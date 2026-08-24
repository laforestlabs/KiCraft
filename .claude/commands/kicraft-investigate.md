---
description: Investigate why a KiCraft run failed — across BOTH the schematic (ERC) and the PCB placement/routing (DRC) — and audit design quality on every run (pass or fail). Locates the run, prints the build verdict, localizes the ERC/DRC failures, classifies systematic code/footprint bugs vs per-design model gaps across all runs, and recommends a generalizable fix. Also audits part-library provenance, BOM realness + the substitution ledger, and LLM thinking-trace wheel-spin. Deliverable: a ranked pipeline-gap report — each finding deduped against known/deferred issues, replay-reproduced on current code, and mapped to its owning module and the gate that should have caught it.
argument-hint: "[KC-XXXXXX board code | uid/pid | pid | /path/to/run] (optional; default: most recent run)"
---

Investigate a failed KiCraft run and hand back a fast, accurate picture of **why** it failed and **whether the fix is generalizable** (a synthesis/layout-code or footprint-library bug that hits *every* design) **or per-design** (this design's model output). Target run: `$ARGUMENTS` (may be empty → most recent run).

**The board is the witness, not the patient.** This skill exists to find **pipeline gaps** — the code/gate/library/prompt changes that improve every *future* board — not to hand-fix this board. A candidate finding is not reportable until it passes §6 (prior-art dedup + replay reproduction on current code) and lands in §7's gap contract with an owning module, the gate that should have caught it, and a verification recipe.

**The artifact-reading engine is `python -m kicraft.cli.triage`** (tested; `tests/test_triage_cli.py` pins it against artifact drift). Do NOT re-implement its readers inline — the previous inline version of this skill rotted silently for weeks. Every subcommand takes the same locator (`KC-XXXXXX | uid/pid | path | empty = latest`) and `--json`.

```bash
REPO=$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/KiCraft"); PY="$REPO/.venv/bin/python"
"$PY" -m kicraft.cli.triage locate "$ARGUMENTS"    # RUN dir + accounts.db row (paste RUN into later steps)
"$PY" -m kicraft.cli.triage run    "$ARGUMENTS"    # the unified failure verdict (start here)
"$PY" -m kicraft.cli.triage audits "$ARGUMENTS"    # design-quality audits (run EVERY time, even rc0)
"$PY" -m kicraft.cli.triage scan                   # cross-run systematic-vs-per-design ranking
```

**Board-id → run-dir resolution (when the user hands you a `KC-XXXXXX` board code).** A board code is *not* a filename or a directory — it is the `projects.board_code` column in `accounts.db`. `triage locate` resolves it to a run dir:

```bash
sqlite3 "$HOME/.kicraft/accounts.db" \
  "SELECT dir_path FROM projects WHERE upper(board_code)=upper('KC-XXXXXX');"
```

→ the run dir (normally `~/.kicraft/projects/<uid>/<pid>`; trust `dir_path` — it may live outside `projects/`). The run dir holds `.kicraft/build.log`, `.kicraft/state.json`, and `generated/<stem>/` (the workspace you hand to `inspect_parent` / `replay` / the power-net snippet).

Gotchas (the previous version rotted here — `resolve_run` in `kicraft/cli/triage.py` only matches the literal `KC-` prefix):

- Pass the **bare code** to every triage subcommand (`KC-6THC46`), never a phrase like `"failed board id KC-6THC46"` — that fails with `could not resolve … under …/projects`. `$ARGUMENTS` frequently *arrives* as natural language; strip it to the `KC-XXXXXX` token before the first triage call.
- A code with no DB row → the build never persisted a board (or the id is wrong). Fall back to `triage locate` with no argument (= most recent run) and read the DB row's `brief`/`stem` to confirm it is the board you were asked about.
- `dir_path` missing on disk → the row exists but the workspace was cleaned; report "board recorded but run dir gone", do not guess a path.


## 0. The map: exit codes, gates, and what `triage run` prints

A `build` is sequential: **synthesize+ERC → place leaves → compose+route parent → verify (DRC) → promote+export**. Route yourself by the failure family `triage run` prints (its `VERDICT:` line; the `build_done` event and the `[build] 4/5 verify:` log line are authoritative over per-round artifacts — a dirty round can be superseded before the promote verify):

| build rc | died at | investigate |
|---|---|---|
| 2 | state schema/read failure | the state.json itself (infra) |
| 3 / 4 | incomplete state / synth input (incl. zero-pin `Mechanical:*` symbols killing stage-prep) | §1 schematic |
| 5 | ERC errors | §1 schematic |
| 6 | placement/compose/route produced **no routable board** — congestion, all-K candidates rejected, degenerate 0-leaf hierarchy, `FreeroutingUnavailableError`, stale board (not produced by this run) | §2 PCB |
| 7 | routed board is dirty — **any of ~8 blockers**: shorts, unconnected, keepout intrusion, gross courtyard overlap, courtyard UNMEASURED (pcbnew absent), form-factor non-conformant, outline-shape non-conformant, `connector_misoriented`, missing component refs | §2 PCB |
| 0 | fab-ready | §3 audits still run |

Warn-only (do **not** fail the build, despite appearing in `rejection_reasons`): `connector_stranded:*`, minor courtyard clips, low utilization / high aspect, `silk_*`. Do not report these as fab-blockers.

**rc6 promotes a partial board on purpose** (no-fallback-previews): `<stem>.kicad_pcb` on disk may be a placed/partial preview. `triage run` reads `<stem>.provenance.json` (`source_kind ∈ routed|placed|partial`, `fresh`) — never judge routing quality from a non-`routed` promote. The replayable seed for an rc6 run is `.experiments/pre_promote_seed.kicad_pcb`.

**Two coordinate conventions:** ERC report `pos` is 1/100 real mm (`triage` already prints real mm); PCB/DRC coords are **already real mm — never ×100 them**.

## 1. Schematic deep-dive (rc ≤ 5)

`triage run` printed the ERC errors; `triage scan` printed each ERC error type's breadth (`>1 design = systematic synthesis-code bug`). Root-cause table:

| ERC error type | Usual root cause | Where to look |
|---|---|---|
| `pin_to_pin` ("Power output … Power output") | PWR_FLAG added to a net already driven by a `power_out` pin | `emitter.py:_power_nets_with_driver`; confirm drivers with the power-net snippet below |
| `power_pin_not_driven` | undriven rail missing PWR_FLAG, or the LLM left the feed pin unwired | below; if the net isn't in `bom.connections` at all → wiring stage (model) |
| `wire_dangling` | trunk router emits a 2-endpoint trunk KiCad doesn't net | `router.py:_draw_trunk` |
| `label_dangling` | net/hier label stub not landing on a wire/pin | `router.py` stub+label fallback / `emitter.py:_emit_root` |
| `pin_not_connected` on one run | wiring stage (LLM) left a pin unwired | `state.json` `bom.connections` — model output |
| "pin missing from netlist" on multi-unit parts | emitter dropped unit-B pins (N4, FIXED) — a fresh hit is a **regression** | `design/synthesis/emitter.py` multi-unit emission |
| rc4 with no retry/park | zero-pin `Mechanical:*` symbol killed stage-prep (N3, FIXED) — fresh hit = regression | stage-prep wiring |

Power-net driver resolution (the one inline snippet kept — a net with a `power_out` pin is driven and must NOT get a PWR_FLAG):

```bash
"$PY" - "<RUN>" <<'PY'
import json, sys
from pathlib import Path
from collections import defaultdict
from kicraft.design.synthesis.symbol_pinout import lookup_pins
from kicraft.design.models import is_power_or_ground_name
run = Path(sys.argv[1])
sf = run / ".kicraft" / "state.json"
if not sf.is_file(): sf = next(run.rglob("state.json"), None)
bom = json.loads(sf.read_text())["bom"]; parts = {p["ref"]: p for p in bom["parts"]}
def ptype(ref, pin):
    p = parts.get(ref)
    if not p: return "?"
    try: pins = lookup_pins(p["symbol"])["pins"]
    except Exception: return "?"
    return next((q["electrical_type"] for q in pins if q["number"] == str(pin)), "?")
nets = defaultdict(list)
for c in bom["connections"]:
    if is_power_or_ground_name(c["net_name"]): nets[(c.get("sheet"), c["net_name"])].append(c)
for (sheet, net), cs in sorted(nets.items()):
    eps = [(ep["ref"], ep["pin"], ptype(ep["ref"], ep["pin"])) for c in cs for ep in c["endpoints"]]
    drv = [f"{r}.{pin}" for r, pin, t in eps if t == "power_out"]
    print(f"  [{sheet}] {net}: {'DRIVEN by ' + ', '.join(drv) if drv else 'no driver -> needs PWR_FLAG'}")
PY
```

## 2. PCB deep-dive (rc 6/7)

`triage run` already localized the failure layer: per-leaf acceptance (with the `no_unconnected` gate detail), the parent round (chosen by **`bool(routed_validation)`**, not the last attempted), `stamp_drc` (shorts>0 PRE-route = the composer stamped overlapping copper), repair evidence, and KRT fingerprints (`backend` / `successful_nets` / `failed_nets` / `input_copper_preservation` in the routing stats).

**(a) Localize a dirty routed board by footprint.** `inspect_parent` re-runs kicad-cli DRC (authoritative — can differ from the persisted `routed_validation`), clusters violations by ref with real-mm coords, and flags packing waste. `--baseline <old report.json>` diffs before/after a replay.

```bash
STEM_DIR=$(find "<RUN>/generated" -maxdepth 1 -mindepth 1 -type d | head -1)
RB=$("$PY" -m kicraft.design.cli_app artifacts --project "$STEM_DIR" --kind routed 2>/dev/null | grep -o '/[^ ]*parent_routed.kicad_pcb' | head -1)
[ -z "$RB" ] && RB=$(find "<RUN>" -name parent_routed.kicad_pcb | sort | tail -1)
OUT=$(mktemp -d); "$PY" -m kicraft.cli.inspect_parent "$RB" --output-dir "$OUT" >/dev/null 2>&1 && sed -n '1,45p' "$OUT/summary.md"
```

A DRC error clustered on one ref across designs (§ scan `clearance footprint refs`) is a **footprint-library** bug — one `.kicad_mod` fix improves every board using it.

**(b) Unconnected nets — the decision order.** Work through these IN ORDER; "add a tie/pour rule" is the *last* resort, not the default story:

1. **Reachability** (was the pad ever escapable?): the leaf block of `triage run` shows `failure_class` — `escape_infeasible` is a **geometry constant** (the escape planner found no legal exit at the board's rule set; re-rolling seeds is worthless; the verdict is honest, not a bug). Check `interface_escapes` for **bare cross-leaf pads** (a leaf lays 0 copper on single-pad interface nets — the dominant rc7 residue class pre-`2d6329e`; kill switch `interface_escape_enabled`). A `no_clear_path` after escapes is a **PLACEMENT bug to fix KiCraft-side** — routing is KRT's job (sole router since `a25e039`); there is no in-house router (C1-v2/A* was scrapped, not shelved).
2. **Budget** (did routing get its time?): `KiCadRoutingTools timed out after Ns` = the route hit `kicad_routing_tools_timeout_s` (default 120) and the attempt failed loudly (a timed-out attempt never yields a board — no partial-output acceptance). `KiCadRoutingTools failed (rc=N)` carries the last 4000 chars of router output. The crash-priced scheduler symptom ("only ONE parent round ran on a big budget" = a crashed round's cost extrapolated the next round over budget) is a scheduler symptom, not placement. Knobs: `kicad_routing_tools_timeout_s`, `kicad_routing_tools_max_iterations`, `kicad_routing_tools_max_ripup`.
3. **Composition** (leaves fine, parent geometry wrong): corridor/mouth misalignment between leaf mouths and the parent channel — dedup to the open **N5b compose mouth-line alignment** workstream before re-reporting. For strand/mouth findings, attribute to the leaf that **set** the edge, not the flagged ref (a foreign leaf's cap poking past another leaf's mouth line is the setter's fault).

The cross-leaf vs not-in-interconnect split `triage` prints uses `interconnect_net_names`; a net **in** it = parent-interconnect failure (seed growth can help), **not in** it = leaf-internal open **or** a bare cross-leaf pad that defeated interconnect *inference* (check `interface_escapes` before blaming the leaf). Artifacts predating the key can't be split — `triage` says so.

**(c) KRT failure signatures** (evidence tiers: `.kicraft/build.log` line → `.experiments/subcircuits/<slug>/debug.json` `routed_validation` + routing stats — `triage run` iterates these debug files):

| signature | meaning | guard / fix |
|---|---|---|
| `KiCadRoutingTools unavailable: …` (preflight) | host misconfiguration: unset `kicad_routing_tools_path`, wrong source version (≠`0.20.2`) / commit (≠`3ceb773…`) / native (≠`0.20.1`), or no sibling `.kicad_pro` | `solve_subcircuits` escalates as ONE hard failure, not per-leaf. Re-provision per `docs/kicad-routing-tools-experiment.md` |
| `KiCadRoutingTools timed out after Ns` | route exceeded `kicad_routing_tools_timeout_s`; process group killed; no board accepted | raise budget or fix placement (reachability first, §2b) |
| `KiCadRoutingTools failed (rc=N): <tail>` | nonzero exit or no output board; last 4000 chars of router stdout/stderr are in the message | read the tail — it names the net/rule that failed |
| `failed to preserve input copper (missing traces=N, vias=M)` | `RoutingCopperPreservationError`: KRT dropped stamped/locked copper | routed output retained at its path for diagnosis — inspect it, never accept the board |
| repair lines (`gnd island repair`, `power strand repair`, `leaf signal repair`) | repair passes DID run — records persist in `routed_validation` (router-independent, still live) | absence of a record ≠ "never ran" only on pre-07-21 artifacts (compactor whitelist) |

Routing stats are `backend: "kicad-routing-tools"` plus `successful_nets` / `failed_nets` / `elapsed_s` / `input_copper_preservation` / `json_summaries` (the first `JSON_SUMMARY` backs the scalar counters).

**(d) Other rc6/rc7 root causes:**

| Symptom | Root cause | Where |
|---|---|---|
| rc6, leaves accepted, empty `routed_validation` every round | KRT can't route the composed parent as placed | `_compose_route` + budget knobs; `_search/_rejected_candidates.json` when all K candidates were rejected |
| rc6 degenerate 0-leaf | BOM chose an all-in-one SoC → architecture collapsed | architecture/BOM partition by IC domain |
| rc7 `stamp_drc.shorts > 0` | composer stamped overlapping copper | `breakout_stubs.py` foreign-pad guard |
| rc7 `illegal_routed_geometry` | usually REAL `clearance`/`copper_edge_clearance` violations — **not** copper outside the outline (that premise was disproven; 5/5 flagged boards had zero outline escapes) | `illegal_geometry_repair` record shows the rip pass verdict |
| rc7 `connector_misoriented:<ref>(mouth …)` | connector mouth not facing its board edge | `cli_app._connector_misoriented` + facings gate; if the part is markerless, the `PCB Edge` Dwgs.User marker is the only reliable opening signal |
| rc7 `form-factor non-conformant` / `outline-shape non-conformant` | delivered geometry violates the captured standard/shape | see §3 intent adherence — distinguish enforcement-off (advisory) from a gate regression |
| rc7 `courtyard_unmeasured` | pcbnew absent at verify → BLOCKING (was a waiver) | build env |
| leaf `place_quality_gate` / `grid_guard=discard_*` | connectivity-first grid assignment rejected/discarded | `leaf_grid_assignment` stats in the leaf's `placement_diagnostics` |

## 3. Design-quality audits — `triage audits`, on EVERY run (even rc0)

Orthogonal defect classes the ERC/DRC gates cannot see. Read each block:

- **[A] library provenance:** all `curated-default`/`kicad-standard` = clean. `home-fetched` recurring across designs = vendor it (`add-part --from-lcsc <C#> --into vendored` + `refresh_sample_previews.py`; corpus-wide view: `python -m kicraft.cli.part_query_report`). `UNKNOWN/MISSING` surviving to a build = resolver/validation hole (`design/cli_app.py` `_unresolved_symbols`/`_unresolved_footprints`).
- **[B] BOM realness:** Pass A `SUSPECT/HALLUCINATED` = a priced C# not in the offline catalog (resolution bug or an online fallback bypassing it). Pass B `MPN-MISMATCH` = real-but-wrong part **candidate** — the matcher already normalizes separators/zero-padding, but verify against the part's role before reporting. Pass C `FABRICATED-LCSC` = a library manifest claims a nonexistent part — re-vendor. **Pass D is the new one:** an MPN deviation from a spec/brief-named part **with an empty `bom.substitutions` ledger** is the `silent_substitution` class (gates §9.23/§9.33 should have forced a ledger entry — name which one missed). The MCU programming-path verdict is deterministic (`mcu_programming_facts`); don't re-derive it by eye, and don't report BOOTSEL+USB (RP2040) or a UPDI pad as "unprogrammable" — §9.29 deliberately accepts those.
- **[C] wheel-spin:** `high_attempts`+`recurring_error` = commit-validation whack-a-mole (often an unwinnable upstream contract). `RECONCILE DEATH` lines = the 2026-07-27 class (`unresolved BOM deficit after N reconcile pass(es)`, byte-identical recommit) — remaining known-deferred: advancing-chain + crystal deterministic-donor. `bom_rounds_maxed`/`tool_loop` = part-lookup thrash (cost driver). Same stuck stage + same error across designs = prompt/validation-contract bug, not a per-design hiccup.
- **[D] intent adherence:** the pipeline NOW captures + enforces mechanical standards (`FormFactor.standard`, form-factor + outline-shape promote gates — enforcement-mode-aware). The verdict distinguishes: standard **not captured** (detection gap) / non-conformant with **enforcement OFF** (advisory gap — invisible to ERC/DRC) / non-conformant while **enforced** (a **gate regression**, headline finding). Shaped boards: ring circumscription uses bbox corners (pessimal for circular content — known-deferred shaped-nesting item). Beyond mechanics, eyeball the BOM/architecture against the brief's named interfaces ("CAN node" → a CAN transceiver? "four mounting holes" → present?).
- **[E] eval/report.json** (self-eval runs): before citing any historical "gate fired" claim, check `observer_rejected` — the judge used to affirm gates whose own evidence self-negated; screened entries are false positives.

## 4. Build log

`<RUN>/.kicraft/build.log` is the primary, authoritative place/route stdout (web AND self-eval runs). The stamp line `[build] code=<sha> branch=<branch>` (runs built ≥ 2026-07-29) dates the code exactly.

```bash
BL="<RUN>/.kicraft/build.log"
tail -80 "$BL"
grep -nEi 'error|traceback|KiCadRoutingTools|segv|exception|unconnected|no_clear_path|ESCAPE INFEASIBLE|Interface escapes|island repair|strand repair|signal repair|code=' "$BL" | tail -50
```

Fallback for deployed web runs with no build.log: `journalctl -u kicraft-web` around the run's mtime. Ignore crawler noise (`/robots.txt`, `/ads.txt`, 404s).

## 5. Cross-run: systematic vs per-design

`triage scan` ranks every failure mode by #designs hit (`>1 = SYSTEMATIC`, fix generalizes; `1` = this design's model output) across the projects dir + self-eval batches. **Read `latest=` and `sha=` before calling anything systematic** — a mode whose last hit predates the owning fix's deploy date is stale evidence; a hit **after** it is a **regression** (headline). Runs without a `sha=` predate the build stamp; date them by `latest=` + the auto-memory fix dates.

## 6. Gate every candidate finding — NEW, LIVE, REPRODUCIBLE?

**(a) Prior-art dedup.** Grep the auto-memory index + plans for the failure signature:

```bash
grep -rli '<signature>' ~/.claude/projects/-home-kicraft-KiCraft/memory/ ~/.omp/agent/sessions/ "$REPO/docs/plans/" 2>/dev/null | head
```

- **KNOWN-FIXED**: all affected runs predate the fix → stale, one appendix line. Any hit after → **REGRESSION**, name the commit.
- **KNOWN-DEFERRED** (current live set: N5b mouth alignment / run_10 GPIO fan-out, GND strand (run_14), reconcile advancing-chain + crystal-donor deaths, shaped-nesting bbox circumscription (run_29)): report "known-deferred, +N runs since <date>", cite the plan/memory, and do **NOT** invent a workaround — masking gates and post-route band-aids are rejected on principle (fix-at-source).
- **NEW** → (b).

**(b) Replay-reproduce on current code — $0, no LLM.** Single-run route verdicts are coin flips (deltas cross grade buckets on identical input) — N-of-3 before claiming a regression; ±3–6 pt judge deltas are noise.

```bash
STEM_DIR=$(find "<RUN>/generated" -maxdepth 1 -mindepth 1 -type d | head -1)
WORK=$(mktemp -d); cp -a "$STEM_DIR" "$WORK/replay"     # NEVER replay in place — replay regenerates .experiments
"$PY" -m kicraft.design.cli_app replay --project "$WORK/replay" --quality good --seed 0
"$PY" -m kicraft.design.cli_app artifacts --project "$WORK/replay"   # honest post-replay verdict
```

Rules that keep replays honest:
- Match `--quality` to the original (`grep 'quality=' <RUN>/.kicraft/build.log`). `--quality fast` **never enters the autoexperiment round loop** — a hook there is silently untested.
- `rm -rf "$WORK/replay/.experiments"` for a cold replay; on **rc6** the replay seed is `.experiments/pre_promote_seed.kicad_pcb` (keep it — the promoted board is a partial).
- `md5sum` the promoted board vs the best-round board when "what actually shipped" matters.
- Replay **cannot verify synthesis-side, parts-library, or LLM-prompt/guardrail changes** (frozen seed, no LLM) — use the offline `synthesize` subcommand for synthesis code, and **§6(b.1) `stage_driver replay`** for wiring/prompt/guardrail changes.
- A/B a fix with the project's `autoplacer.json` **kill switch**, same code both sides, and measure both sides in **ONE** script after one replay each — never compare artifacts across separately-scripted replay runs.
- Never DRC/validate a board copied without its `.kicad_pro`/`.prl`/`*_autoplacer.json` — bare copies get default netclass rules stamped in and manufacture fake violations.
- Bisection is legitimate (a FreeRouting-era loop-hang was once pinned by bisecting 31 locked wires to one segment).

**(b.1) LLM-stage replay — prompt / guardrail / wiring changes (live LLM, budget-capped $0.25).** The deterministic `cli_app replay` above can't exercise the LLM design stages, so a prompt or commit-validation change was previously unverifiable — it fell to a human. `stage_driver` now drives those stages live against the run's frozen `state.json`:

```bash
# Re-run ONE LLM stage (e.g. the failed wiring stage) from a frozen state.json
"$PY" -m kicraft.server.stage_driver replay \
  --state "$RUN/.kicraft/state.json" --stage wiring --budget 0.25

# Full end-to-end: a brief through ALL LLM stages + the deterministic build
"$PY" -m kicraft.server.stage_driver run \
  --brief "<brief text>" --workspace "$(mktemp -d)" --budget 0.25 [--no-build --quality good]
```

Rules:
- `replay` copies `state.json` into a fresh temp workspace (the run dir is **never** mutated) and recovers the brief from `intent.goal` (fallback `brief.txt`). Output: `[ok/FAIL] <stage> cost=… attempts=…` then the workspace + state path. On `ok`, inspect `<workspace>/.kicraft/state.json` `bom.connections` to confirm the netlist is *correct*, not merely gate-passing (e.g. a pull resistor's rail pin really landed on `+3V3`/`GND`).
- `run` drives `intent→functional_spec→architecture→bom→wiring` then `cli_app build` (skip with `--no-build`). This is the harness for "does the whole pipeline still work after a change" — run a known-good brief, not the regression target.
- The budget (`--budget`, default $0.25) is a **hard per-run cap**: `_BudgetGuard.preflight()` refuses before any completion that would cross it, on top of the global daily/total ceilings. Granularity is one completion.
- The API key loads from `$REPO/.env` via `Settings.from_env()` — run with cwd = the repo (or export `OPENROUTER_API_KEY`); `KICRAFT_LLM_MODE=mock` runs the same path at $0 (fails at the mock transcript if none is recorded).
- LLM verdicts are stochastic (temperature > 0): **N-of-3** before claiming a prompt/guardrail change fixed or regressed anything — a single `ok`/`FAIL` is a coin flip, exactly like the route verdicts in §6(b).

**(c) Name the gate that should have caught it.** For every defect that survived past its origin stage: which deterministic gate (synthesis 9.x checks, wiring normalizers, leaf acceptance, composer stamp-DRC, promote verify incl. the form-factor/outline/facings gates, review clamp) could have caught it earliest, and why did the existing one miss (fail-open on None? warn-only? bbox-based and rotation-blind?). "Extend gate X to catch Y at stage Z" is the most common shape of a shipped fix; a finding with no gate answer is under-investigated.

## 7. Report — the ranked pipeline-gap contract (the deliverable)

Rank by breadth (scan #designs) × recency (`latest=`/`sha=`) × severity (fab-blocking > silently-wrong-board > quality > cost). Top 3 gaps max; everything else one appendix line. Every ranked gap fills all six fields:

```
GAP <n>: <one-line name>                [code | footprint-library | gate-hole | prompt/contract | infra]
  evidence:  N/M designs, latest <date> — <≤4 run ids>; if N==1: replay-verified? y/n
  detect:    earliest stage/gate that could have deterministically caught it + why the current one missed
  source:    <file:func> — the single point that sets the bad value; fix THERE, never a downstream mask
  fix:       <the one change>; guard: <the test that keeps it fixed>
  verify:    replay <run(s)> → expect <specific delta, e.g. unconnected 2→0, rc7→rc0>
  prior-art: NEW | REGRESSION of <commit/memory> | KNOWN-DEFERRED <plan/memory> (+N runs since)
```

After the gap list: one paragraph per-run verdict (failing stage, specific failure, right coords) and the §3 audit findings **even when the build passed**. Pure per-design model output goes in the appendix — unless the same mistake recurs across designs (then it's a prompt/contract gap and ranks).

## 8. Headless mode (`KICRAFT_INVESTIGATE_HEADLESS=1` — the /admin/support runner)

- **Budget: ~25 min hard** (the runner kills at 30). Skip §6b replay for anything dense (>10 leaves or a >600s original route budget); mark those findings `PLAUSIBLE (replay not run — headless budget)` and include the exact replay command in the report so a human can run it.
- **Only your final message survives** (`omp -p` keeps the final assistant message; mid-run notes are discarded). The full §7 report — gap blocks, per-run verdict, audit findings — must be in that one final message, self-contained, no references to "above".
- **Never launch a background replay or promise "I'll report back"** — the session ends with your final message and anything still running dies with it. Replay synchronously inside the budget, or skip it and mark the finding PLAUSIBLE with the exact command.
- No user is present: never ask questions; make the conservative call and record the uncertainty in the report.
- Stay read-only outside tempdirs: replay copies and `mktemp -d` outputs only; never modify the run dir, the repo, or memory.
