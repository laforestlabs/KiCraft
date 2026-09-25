# Next session — KiCraft live debugging (written 2026-09-25, end of the parts-research session)

Paste this as the first message of the next session. It assumes no memory of the previous one.

---

Work in `/home/kicraft/KiCraft` on the production box (the live site is https://kicraft.io,
NiceGUI on `127.0.0.1:8080`, Caddy in front; services run as detached processes managed by
`deploy/restart-web.sh` and `deploy/restart-build-worker.sh` — see `AGENTS.md`).

## Read these first, in this order

1. `docs/plans/live-debug-skill-plan-2026-09-25.md` — the owner's living plan: what they want,
   what has been fixed, what is still open, and the loop/plain-language rules.
2. `~/.kicraft/debug/surprise-40-3v3-converter-20260925/.kicraft/debug/findings.json` — the
   running record of the last run: 16 gaps with evidence and validation, plus
   `assumptions_taken` (every default the run took on the owner's behalf) and `run-summary.json`.
3. `.agents/skills/kicraft-debug/SKILL.md` — the debugger: interactive and loop modes,
   *Default, then analyse*, *Say it plainly*, *Keep the owner oriented*, the stop conditions.

## Where things stand

- The five design steps of the seed-40 brief (`5V_3V3_CONVERTER`: host 5 V header in, reverse
  polarity protection, 3.3 V linear output, power LED, four-layer stack-up) are committed in that
  workspace and the deterministic build exits 0: 0 shorts, 0 unconnected, 8 parts, four copper
  layers, four copper gerbers.
- Eight commits are pushed on `simplify/bom-wiring-pipeline` (`1a79975..40b9a1f`), the last two
  being the part-research capability and the two selection seams it exposed.
- The pipeline can now do things it could not this morning: deliver the copper stack a brief asks
  for, research a demanded part class itself (offline catalog search → vendor into the
  machine-wide parts library → reviewed record, once), see every reviewed part the library holds,
  size a connector to the count the brief states, and keep a LED's series resistor.
- Services were restarted at the end of the session and verified: web HTTP 200, build worker
  ready. The live site is running this code.

## Open work, most valuable first

1. **Researched parts carry no ratings, on purpose.** `part_research.build_record` leaves
   `operating_limits` empty because no rating was read from a datasheet, so rating gates treat
   those parts as unproven. Next step: fill the limits this loop *can* verify (the catalog's own
   parametric description, the package, the pin count) and mark anything it cannot as unverified
   rather than absent — or add a review step a human confirms once. This is the difference
   between "the part exists and is in stock" and "the part is right for this rail".
2. **The class claim is a description match.** `choose_candidate` accepts a row because the
   catalog's phrase matched its text. A second, independent signal would raise confidence: the
   package and contact count against the demand, the parametric fields, or a typed yes/no from the
   decision model (the pipeline already has a Jev client in `kicraft/server/decision_layer.py`).
3. **The regression evidence still counts only two copper layers**
   (`kicraft/eval/geometry_artifact_evidence.py`), so a four-layer board's inner planes are
   invisible to a self-eval sweep. Small change, but it changes what the rubric measures — the
   owner's call (plan item 16).
4. **The layout search reports three failed rounds and `Best score: 0.00` immediately before
   promoting a clean board** (finding `P12`). Report the accepted candidate's own round and score,
   so a successful build never reads as three failures.
5. **Internal block net names reach the delivered board**: `__KICRAFT_RECIPE__…__gate` and
   `__lowerer__led__led_anode` are the net names on the shipped schematic (finding `P13`).
   Humanise them (`PROTECTION_GATE`, `LED_ANODE`).
6. **Protection topology is the owner's choice**: the shipped design uses the reviewed P-channel
   MOSFET block; a series Schottky is now buildable from the two reviewed order codes that were
   invisible before (B340A C64982, SS34 C8678). Cheaper and simpler if they prefer it.
7. **No gate enforces a stated protection requirement** (finding `P2`, owner decided to leave it
   that way). Revisit only if a board ever ships without the protection its brief asked for.
8. **Run the loop on more seeds.** Loop mode exists for exactly this: one workspace per seed under
   `~/.kicraft/debug/`, self-checked review, per-step saving, an end-of-run summary and a
   machine-readable last line. Expect new gaps; record each one with its evidence and the test
   that fails without the fix.

## How the owner wants this work done

- **Plain language.** No internal vocabulary in reports ("facet", "slot", "candidate",
  "obligation", "recipe", "payload" are all banned): name the concrete thing — the sentence in the
  brief, the part, the voltage, the pin.
- **Frequent short progress notes.** What just happened, what it means, what happens next, after
  every state change — never one report at the end of a long run.
- **Default, then analyse.** An unknown the design can absorb is defaulted the way the live
  product would, with its cost analysed and recorded; only an unabsorbable unknown stops the run.
- **Loop mode when they are not watching, interactive when they are.** Interactive: one piece of
  the answer at a time, and only an explicit "accept" saves a step.
- **Fix the pipeline, not the answer.** Every generalizable gap gets a patch at the source and the
  smallest test that fails without it. Never hand-edit `.kicraft/state.json`.
- **Commit and push** when a batch is done; deploy with `./deploy/deploy-production.sh` and read
  its health line. Never claim a run or a build finished unless it did.

## Handy commands

```bash
# the walkthrough (interactive), from a workspace directory
/home/kicraft/KiCraft/deploy/run-with-env.sh kicraft-stage-debug debug-draft --workspace . \
  --stage <intent|functional_spec|architecture|bom|wiring> \
  --brief-file /tmp/kicraft_debug_brief.txt --budget 0.25
/home/kicraft/KiCraft/deploy/run-with-env.sh kicraft-stage-debug debug-commit --workspace . \
  --stage <stage> --history-message-file /tmp/kicraft_debug_history.txt

# part research, by hand (also runs automatically inside the architecture and parts steps)
/home/kicraft/KiCraft/deploy/run-with-env.sh kicraft research-part <class> --json

# the build, and the deploy
/home/kicraft/KiCraft/deploy/run-with-env.sh kicraft build .kicraft/state.json generated --quality good
/home/kicraft/KiCraft/deploy/deploy-production.sh
```

## First action

Read the plan and the findings, then either take the top open item (ratings for researched parts)
or start a loop run on a new seed and report the first gap you find — in plain words, with the
evidence beside it.
