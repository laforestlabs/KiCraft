# Autoexperiment round loop → `RoundScheduler` refactor

**Status:** planned (this doc is the design). Written 2026-07-13 against `484620e`.
**Goal:** stop accreting `if`-plus-mutable-flag special cases in `main()`'s round loop.
Every scheduling decision moves into one small, unit-testable object; the loop body
becomes pure mechanism. Net effect: code REMOVED from `main()`, not added beside it.

## Why now

`kicraft/cli/autoexperiment.py::main()` runs one `for round_num in range(...)` loop
(~2461–3290, ~830 lines) that mixes two very different kinds of code:

- **Mechanism** — run the leaf solve subprocess, compose the parent, score, write
  status/artifacts. This is the bulk and it's fine where it is (long, but linear).
- **Policy** — decide whether to run another round, and with what restrictions.
  This is the part that has been growing an `if` + loop-scoped mutable per fix.

Current policy inventory (each one is state threaded through the loop body):

| Policy | State | Where |
| --- | --- | --- |
| Stop request | — | `_check_stop_request` at loop top + after loop |
| Wall budget (WS2) | `max_wall_s`, `ema_round_s` | gate at ~2467, EMA update at ~3289 |
| Unroutable early-abort | `unroutable_streak`, `abort_rounds` | mid-loop, returns `_RC_LEAF_UNROUTABLE` |
| Quality-rejection streak (WS2) | `quality_streak`, `quality_abort_rounds` | mid-loop `break` |
| Parent capout streak (WS2) | `parent_capout_streak` | mid-loop `break` |
| Keep/best promotion | `best_score`, `kept_count`, `_best_config` | tail of loop |
| N2a wall-budget rescue round | `wall_rescue_only`, `wall_rescue_leaf_s`, `wall_rescue_attempted` | **reverted** — see below |

The N2a rescue round was implemented, reviewed, and pulled back out precisely because
it was the seventh special case: three more mutables plus an env-var clamp woven into
the budget gate. The working implementation (with its tests) is preserved verbatim in
`docs/plans/patches/n2a-wall-rescue.patch`; it becomes a ~15-line scheduler policy here.

None of these policies is unit-tested today as a policy — they are only exercised by
integration tests that run real subprocesses, so a budget-arithmetic bug costs a
multi-minute repro. That, plus the accretion rate, is the case for the refactor.

## Target shape

New module `kicraft/cli/_round_scheduler.py` (follows the `_compose_*.py` split
convention). Three pieces, all plain dataclasses/methods, no subprocess, no I/O
except reading leaf-acceptance state that the caller passes in:

```python
@dataclass(frozen=True)
class RoundPlan:
    round_num: int
    seed: int
    only: list[str]            # leaf selectors; [] = all (args.only pass-through)
    leaf_deadline_s: float | None  # None = inherit the configured deadline
    note: str                  # human-readable "why this plan" for the build log

@dataclass(frozen=True)
class RoundOutcome:
    duration_s: float
    solve_rc: int
    parent_route_rc: int
    score: float
    leaf_accepted: int
    leaf_total: int
    structural_unroutable: list[str]   # leaf selectors this round
    quality_fail: dict[str, str]       # leaf -> rejection signature
    parent_route_capped: bool          # FreeRouting hit its timeout cap
    unpinned_leaves: list[str]         # leaves with NO accepted artifact on disk

class RoundScheduler:
    def plan_next(self, *, elapsed_s: float, stop_requested: bool) -> RoundPlan | Finalize
    def observe(self, outcome: RoundOutcome) -> None
```

`Finalize` is a tiny frozen dataclass `(reason: str, rc_hint: int | None)` so the
unroutable-abort path can still map to `_RC_LEAF_UNROUTABLE` while every other
finalize keeps the best-so-far grading path. `main()`'s loop collapses to:

```python
scheduler = RoundScheduler(rounds=args.rounds, max_wall_s=max_wall_s, ...)
while True:
    decision = scheduler.plan_next(
        elapsed_s=time.monotonic() - start_ts,
        stop_requested=_check_stop_request(work_dir),
    )
    if isinstance(decision, Finalize):
        print(f"[scheduler] {decision.reason}")
        break
    outcome = _run_round(decision, ctx)   # the existing loop body, verbatim
    scheduler.observe(outcome)
```

All seven policies become private methods on the scheduler, each a few lines of
arithmetic over state that `observe()` maintains (EMA, streak dicts, rescue flag,
round count). The rescue policy in particular:

```python
# inside plan_next(), when the budget gate would otherwise finalize:
if (not self._rescue_attempted and not self._only_locked
        and remaining >= _WALL_RESCUE_MIN_S and self._last_unpinned):
    self._rescue_attempted = True
    return RoundPlan(..., only=self._last_unpinned,
                     leaf_deadline_s=max(60.0, remaining * 0.6),
                     note=f"rescue round for {self._last_unpinned}")
```

### What stays out of the scheduler (deliberately)

- **Config mutation / search** (`_mutate_config`, `_random_sample_config`, the
  feasibility floor). This is *search* policy, not *scheduling* policy, and it is
  already reasonably factored into free functions. If it ever grows another mode,
  extract a sibling `ConfigSearch` object — do not fold it in here.
- **Live-status writes, artifact/round-dir management, scoring.** Mechanism; stays
  in `_run_round`.

### Deadline plumbing (kill the env-var clamp)

The patch clamps the rescue leaf deadline by mutating `KICRAFT_LEAF_SOLVE_MAX_WALL_S`
in the child env. With `RoundPlan.leaf_deadline_s` as an explicit field, pass it as
an explicit `--max-wall-s` flag on the `solve_subcircuits` command line in
`_build_solve_cmd` (adding the flag to solve_subcircuits if it only reads the env
today, keeping the env as fallback). Explicit beats ambient.

## Migration steps (each lands green on its own)

1. **Extract `_run_round(plan, ctx)`** — mechanical move of the loop body into a
   function returning `RoundOutcome`. `ctx` is one dataclass holding the loop's
   read-only locals (paths, args, status writers). No behavior change; existing
   integration tests are the check. *This is the only risky step — do it as a pure
   move, no edits, so the diff reviews as indentation.*
2. **Introduce `RoundScheduler`** with the existing policies only (round count,
   stop, wall-budget EMA, the three streaks, keep/best promotion feeding
   `_best_config`). Delete the corresponding loop-scoped mutables from `main()`.
   Add unit tests driving the scheduler with synthetic `RoundOutcome`s — budget
   exhaustion, streak aborts, EMA math — no subprocesses, milliseconds each.
3. **Port the N2a rescue policy** from `docs/plans/patches/n2a-wall-rescue.patch`
   (logic + its two `_unpinned_leaf_selectors` tests, which move nearly verbatim —
   the selector helper itself stays in autoexperiment.py or moves to the scheduler
   module). Delete the patch file in the same commit.
4. **Explicit deadline flag** replacing the env clamp (see above). Small, separate.

## Acceptance criteria

- `main()` contains **zero** scheduling `if`s and zero loop-scoped policy mutables;
  the driver loop is ≤ ~15 lines.
- Every policy has a unit test that runs without subprocesses.
- Behavior parity: `kicraft replay` of a frozen run (e.g. the 20260710 batch's
  run_09) produces an identical round sequence and verdict before/after step 2.
- `docs/plans/patches/n2a-wall-rescue.patch` is gone (absorbed), and the rescue
  fires on a synthetic budget-starved outcome in the unit suite.

## Non-goals

- No changes to solve/compose/scoring semantics, artifact layout, or `rounds/*.json`
  payloads (three processes depend on that shape — see CLAUDE.md).
- No attempt to make the scheduler configurable/pluggable. One class, concrete
  policies, in the order they short-circuit. Simple beats general here.
