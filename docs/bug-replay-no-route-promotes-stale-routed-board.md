# BUG: `replay --no-route` promotes a STALE routed board, not the fresh placement

> **RESOLVED.** Fixed holistically in `kicraft/cli/artifact_paths.py` (intent-based
> resolver: `kind="placed"` never returns a routed board) + a run-scoped freshness
> gate at the promote sites + honest promote mtime (`shutil.copy`) + run-id
> provenance + the `kicraft artifacts` query. The canonical contract now lives in
> **`docs/ARTIFACTS.md`**. This file is kept for the investigation history.

**Severity:** high (silent) — it makes the codebase's own deterministic placement-iteration
tool show a board the current run did not produce. It cost a multi-hour debugging session
where every code change and even brute-force overrides appeared to have "no effect."
**Component:** `kicraft/design/cli_app.py` (`_cmd_replay` / `_find_placed_parent`)
**Status:** open — write-up for a separate agent to fix.

## Summary

`kicraft replay --no-route` is documented as the fast, fully-deterministic placement-only
mode whose "footprint positions are the determinism guarantee we assert." In reality, when
the project directory already contains a `parent_routed.kicad_pcb` from any earlier routed
run, `--no-route` **promotes that stale routed board** as if it were the current run's
placement. The current run's *fresh* placement is computed and written, then ignored.

Because the promote uses `shutil.copy2` (which copies mtime), the promoted file even keeps
the **old timestamp**, so nothing looks updated and the staleness is invisible.

## Repro

```bash
WS=<a synthesized workspace that has been routed at least once>   # has .../parent_routed.kicad_pcb
# 1. note D1 (or any footprint) position in the promoted board
kicraft replay --project "$WS" --quality fast --no-route --seed 0
# 2. make ANY placement change (edit the solver, or just re-run) and repeat
kicraft replay --project "$WS" --quality fast --no-route --seed 0
# -> promoted <stem>.kicad_pcb is byte-identical every time; its mtime never advances,
#    matching the OLD parent_routed.kicad_pcb mtime exactly.
```

Observed during the KC-VTRVY7 investigation (`~/.kicraft/projects/1/90/generated/5X9_ARRAY_OF`):
forcing the array block to every rotation (0/90/180/270) **and** translating it +500 mm at
the end of `PlacementSolver.solve()` produced an identical promoted board each time; the
promoted board and `parent_routed.kicad_pcb` shared the exact mtime `16:19:38.243574772`
across ~15 runs spanning 35 minutes.

## Root cause

In `kicraft/design/cli_app.py`:

1. `_cmd_replay`, `--no-route` branch (~`:2988-2998`):
   ```python
   placed = _find_placed_parent(project_dir)
   ...
   shutil.copy2(placed, pcb)          # copy2 PRESERVES source mtime
   print("[build] 3/5 promoted placed parent -> ...")
   ```
2. `_find_placed_parent(project_dir)` returns **the routed board first**:
   ```python
   routed = _find_routed_parent(project_dir)
   if routed is not None:
       return routed                  # <-- stale parent_routed.kicad_pcb wins
   # only if NO routed board exists does it fall back to the fresh
   # parent_pre_freerouting.kicad_pcb that --no-route actually produced
   ```

So on any previously-routed project, `--no-route` returns the old routed board and never
looks at the placement the current run just made.

### Where the current run's fresh placement actually goes

The fresh, current-run parent placement IS written — to
`<ws>/.experiments/subcircuits/<parent>/parent_pre_freerouting.kicad_pcb` (the stamp runs in
a subprocess and also preserves mtime, so this file likewise looks "old" by timestamp but its
*content* is current). The per-candidate boards
(`.../<parent>/_search/cand_NN.kicad_pcb`) are written and then `rmtree`'d after the search
(`kicraft/cli/compose_subcircuits.py`, the candidate-search cleanup ~`:2485`).

Verified: measuring `parent_pre_freerouting.kicad_pcb` (instead of the promoted board) DID
reflect code changes — a clean A/B showed two different placements (md5 `0752699f` vs
`fb0dca52`), proving the placement pipeline is live and only the *promotion/measurement* was
stale.

## Why it's nasty

- `shutil.copy2` preserving mtime hides the staleness — the usual "did the file change?"
  check (mtime / `find -newer`) says "no", which *looks* like determinism but is actually a
  stale copy.
- `--no-route` is exactly the mode people reach for to iterate on placement code, so the
  failure mode is "my placement change does nothing," which sends you debugging the wrong
  layer (the solver) instead of the tool.

## Proposed fixes (pick one; the agent should decide)

1. **Prefer the fresh pre-freerouting board in `--no-route`.** `_find_placed_parent` is
   correct for *routed* replays (routed-board-first is what you want there), but `--no-route`
   should explicitly fetch the `parent_pre_freerouting.kicad_pcb` the current run produced —
   not whatever routed board happens to be on disk. Simplest: give `--no-route` its own
   resolver, or pass a `prefer_pre_freerouting=True` flag through `_find_placed_parent`.
2. **Freshness guard.** If the chosen board's mtime predates the start of this replay (capture
   a start timestamp), refuse to promote it and error loudly: "stale board; the current run
   produced no fresh parent board" — never silently promote a board older than the run.
3. **Use `shutil.copy` (not `copy2`) for the promote**, or `os.utime(pcb)` after copy, so the
   promoted board's mtime reflects when it was promoted. This alone doesn't fix the staleness
   but removes the mtime-masking that makes it invisible. (Do this *in addition to* 1 or 2.)
4. **Clean stale outputs at the start of a replay** (delete the previous
   `parent_routed.kicad_pcb` / promoted `<stem>.kicad_pcb` before re-running) so a fresh run
   never inherits a previous run's artifacts.

Recommended: **1 + 3** (correct source + honest mtime), with **2** as a cheap safety net.

## Acceptance test

On a previously-routed workspace, with `--no-route`:
- Make a deterministic placement change (e.g. bump a solver weight) and confirm the promoted
  board's footprint positions change.
- Confirm the promoted board's mtime advances on every run.
- Confirm `replay --no-route` never promotes a board older than the run's start time.

## Cross-reference

See `docs/kc-vtrvy7-placement-verification-trap.md` for the investigation this surfaced from.
