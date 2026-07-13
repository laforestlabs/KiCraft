# KiCraft board artifacts — the source of truth

This is the canonical answer to the two questions that keep biting: **where is the
current board on disk, and was it produced by this run (or is it stale)?**

If you take one thing from this doc: **don't glob `.experiments` by hand and don't
measure `<stem>.kicad_pcb` for placement work. Run `kicraft artifacts`** (or call
`kicraft/cli/artifact_paths.py`). One resolver, one provenance record, one truth.

```bash
kicraft artifacts --project <synthesized-project-dir> --kind all      # human
kicraft artifacts --project <dir> --kind placed --json                # machine
```

## On-disk layout

A synthesized project keeps every board the layout engine produces under
`<project_dir>/.experiments/subcircuits/<slug>/`. The top-level `<stem>.kicad_pcb`
is a **promoted copy** of the best board a build/replay reached — never edit or
measure it as if it were the engine's output.

```
<project_dir>/
├── <stem>.kicad_pcb            # PROMOTED copy of the best board this run reached
├── <stem>.provenance.json      # which run promoted it, from what source, md5, fresh?
├── <stem>.kicad_sch / .kicad_pro / .kicad_prl
├── .experiments/pre_promote_seed.kicad_pcb   # full-component seed saved before the
│                               # promote clobbers <stem>.kicad_pcb; replay restores it (rc6)
└── .experiments/subcircuits/
    └── <slug>/                 # one dir per leaf or parent subcircuit
        ├── metadata.json       # parent: carries run_id + generated_at (routed compose only)
        ├── debug.json, solved_layout.json
        ├── parent_pre_freerouting.kicad_pcb   # PLACED parent (what --no-route produces)
        ├── parent_routed.kicad_pcb            # ROUTED parent (post-FreeRouting)
        ├── leaf_pre_freerouting.kicad_pcb     # PLACED leaf
        ├── leaf_routed.kicad_pcb              # ROUTED leaf
        ├── leaf_illegal_pre_stamp.kicad_pcb   # legality-REJECTED placement (rc6 preview only)
        ├── round_NNNN_*.kicad_pcb             # per-round snapshots (autoexperiment)
        ├── _search/cand_NN.kicad_pcb          # candidate placements (rmtree'd after the winner)
        └── renders/*.png
```

Names are defined **once**, in `kicraft/cli/artifact_paths.py`
(`PARENT_ROUTED`, `PARENT_PLACED`, `LEAF_ROUTED`, …). Do not hard-code these
literals in new code — import them.

## Resolving a board (intent-based)

`kicraft/cli/artifact_paths.py` is the only resolver. It is **intent-based**, not
"richest board wins":

| Want | Call | Returns |
|---|---|---|
| the routed parent | `resolve_parent_board(dir, kind="routed")` | `parent_routed.kicad_pcb` or `None` |
| the **placed** parent | `resolve_parent_board(dir, kind="placed")` | `parent_pre_freerouting.kicad_pcb` only — **never** a routed board |
| an rc6 preview leaf | `resolve_best_leaf_board(dir)` | richest leaf (routed > placed > rejected) |

`kind="placed"` **never** falls back to the routed board. That is the fix for the
old `replay --no-route` trap, where the resolver returned a *routed* board from a
previous run and the placement-only run silently promoted it. When several parent
dirs exist, the newest is chosen by board/metadata **mtime** (deterministic), not
`iterdir` order or alphabetical `sorted()[-1]`.

## Freshness & provenance — "did this run produce it?"

Every command that promotes a board (`build`, `replay`, `manual-route`) calls
`artifact_paths.ensure_run_context()` at entry, which sets two env vars inherited
by the compose/route subprocesses:

- `KICRAFT_RUN_ID` — a short id for this run (honors one the web driver injected;
  same convention as `parts_library/query_log.py`, so board provenance and
  part-lookup logs correlate).
- `KICRAFT_RUN_STARTED_AT` — wall-clock start, the freshness reference.

The promote tail **gates on freshness** (`produced_by_this_run`): a board the
current run did not produce is **never silently promoted**.

- **Routed fab promote** and **`--no-route` placement promote**: if the resolved
  board isn't from this run → **loud failure** (the routed case falls through to
  the rc6 inspection preview; the `--no-route` case errors `rc6`). No more "my
  change had no effect" because a stale board was copied.
- **rc6 best-partial preview**: this path *exists* to show "whatever this run
  got" on failure, so a non-fresh partial is a **warning**, not a hard error —
  the board is still shown for inspection.
- **Leaf *inputs* are never gated.** `replay` and the autoexperiment
  `--parents-only` phase reuse frozen leaves from a prior run on purpose.

`produced_by_this_run` trusts a **positive `run_id` match** (authoritative; immune
to clock skew and mtime-preserving copies) and otherwise falls back to
`mtime >= run_started_at`. It never *rejects* on a `run_id` mismatch, because
`metadata.json` (the run_id source) is only rewritten on a **routed** compose —
on a stamp-only `--no-route` run the board is freshly `Save()`'d (fresh mtime) but
`metadata.json` is not, so mtime is the correct signal there.

Promotion uses `shutil.copy` (not `copy2`), so `<stem>.kicad_pcb`'s mtime reflects
**when it was promoted** — `find -newer` / "did it change?" work again.

`<stem>.provenance.json` records the last promote: `run_id`, `source_board`,
`source_kind`, `md5`, `fresh`. It is the authoritative, agent-facing record of
what the on-disk `<stem>.kicad_pcb` actually is.

## Invariants (do not break)

- **Provenance lives only in JSON sidecars — never inside a `.kicad_pcb`.**
  Boards are byte-stable; the geometry goldens (`scripts/replay_corpus.py`,
  `tests/test_replay_command.py`) compare footprint x/y/rotation only.
- **Never measure `<stem>.kicad_pcb` for placement A/B.** Resolve the placed board
  (`kind="placed"` / `kicraft artifacts --kind placed`). The A/B scripts
  (`replay_corpus.py`, `ab_compose.py`) already do, and follow the
  "delete the canonical output → compose → measure" discipline.
- **`.experiments` is not wiped at run start.** Staleness is handled by the
  freshness gate, which lets `replay` reuse frozen leaves and preserves the rc6
  preview. Don't add a blanket wipe.

## See also

- `kicraft/cli/artifact_paths.py` — the implementation and the precedence rules.
- `docs/bug-replay-no-route-promotes-stale-routed-board.md` — the investigation
  this resolves (kept for history).
