# KiCraft skill-eval rubric

> **Rubric v1 — `sha256:df90130898218fba9dd75e19af5f6703b641c14607303c4e7bcbfc9948a6e5e9`**
>
> Human-readable mirror of [`rubric.yaml`](./rubric.yaml), which is the canonical
> source of truth. If the two ever disagree, **`rubric.yaml` wins.** Regenerate the
> stamp above with `bin/rubric_hash.py compute` after any scoring change.

This rubric scores a single **KiCraft run** — one subject `claude` session
that took a PCB brief through the pipeline. It does **not** score PCB-layout
quality (that is `kicraft/scoring/`, a different thing).

A score is **only comparable to other scores carrying the same `sha256`.** When
the rubric changes, the hash changes, and old scores move to a separate cohort.

## Scoring model

- **10 dimensions**, each scored on an anchored **0–4 level**.
- Two classes: **Class C** (deterministic, scored by `bin/score_run.py` from
  machine signals — the reproducible baseline) and **Class J** (judgment, scored
  by the observer agent reading the transcript + artifacts against the stage
  specs and sound EE practice — the "gotcha" layer).
- Weights sum to **100** (50 C / 50 J), so a flawless run scores 100:

  `weighted = Σ (weight_i × level_i / 4)`

- **Hard-fail gates** then cap the result: `final = min(weighted, lowest triggered cap)`.
- The final score maps to a **grade + ship verdict**.

## Class C — deterministic (50 pts, scored by `score_run.py`)

| Dimension | Wt | What it measures | Signal source |
|---|---|---|---|
| `pipeline_completion` | 12 | How far it got, did it land | state.json slots, `synthesis_check.status` |
| `computing_error_cleanliness` | 16 | ERC errors, failed checks, crashes | `*_erc.rpt`, `synthesis_check.failed_checks`, transcript errors |
| `convergence_efficiency` | 8 | Clean convergence vs thrash | re-commits / aborts / retries (history + transcript) |
| `latency` | 6 | Wall-clock start→synthesized | transcript timestamps (fallback: history → `checked_at`) |
| `interaction_friction` | 8 | Questions + permission prompts vs expectation | transcript, `settings.local.json`, scenario band |

**Anchor highlights** (full text in `rubric.yaml`):

- `computing_error_cleanliness`: **4** = 0 errors / 0 failed checks / 0 warnings;
  **3** = clean but warnings present; **1** = 1–10 ERC errors or ≥2 failed checks;
  **0** = synthesis-blocking crash or >10 ERC errors.
- `convergence_efficiency`: **4** = clean single pass (user-driven re-commits don't
  count); penalties only for the agent's *own* error-driven re-commits.
- `interaction_friction`: scored **relative to the scenario's
  `expected_question_band`** — asking on an ambiguous brief is good; interrogating
  a clear one, or skipping a needed interview, is bad. Permission prompts above
  the documented allowlist floor are always friction.

## Class J — judgment (50 pts, scored by the observer)

| Dimension | Wt | What it measures |
|---|---|---|
| `spec_compliance` | 10 | Obeyed SKILL.md + stage specs (no state hand-edits, no sub-agent `Read`, no `cd`, no silent substitution) |
| `intent_fidelity` | 10 | Output honors stated constraints/preferences |
| `electrical_soundness` | 16 | **The gotcha dimension** — grounding/ground-loops, decoupling, MCU programming path, protection, strap resistors, thermal, rail sizing |
| `part_selection_quality` | 8 | Right part, right source, sane footprint/symbol, ratings fit |
| `failure_honesty` | 6 | Surfaced problems vs "looks healthy, isn't"; stopped cleanly on hard failure |

`electrical_soundness` carries the heaviest single weight on purpose: a schematic
can be ERC-clean (high Class-C) and still be a bad circuit. This is where the
observer earns its keep — checking the design, not just the files.

## Hard-fail gates

A triggered gate caps the **final** score regardless of the weighted sum; the
lowest cap wins. Script-detectable gates are set by `score_run.py`; observer
gates are set when grading the relevant dimension.

| Gate | Detected by | Cap | Condition |
|---|---|---|---|
| `synthesis_broken` | script | 25 | synthesize crashed / produced no files |
| `erc_errors` | script | 45 | ≥1 ERC **error** (warnings don't trigger) |
| `unprogrammable_mcu` | observer | 50 | MCU present, no first-flash path, gap not surfaced |
| `silent_substitution` | observer | 55 | inferior part swapped in unsurfaced, reached synthesis |
| `state_corruption` | observer | 60 | `state.json` hand-edited, bypassing stage-commit |

## Grade bands → verdict

| Final score | Grade | Verdict |
|---|---|---|
| ≥ 90 | A | SHIP |
| 75–89 | B | SHIP-WITH-FIXES |
| 60–74 | C | REWORK |
| 40–59 | D | NOT-READY |
| < 40 | F | BROKEN |

## Changing the rubric

1. Edit `rubric.yaml` (weights, anchors, gates, bands).
2. Bump `meta.version`.
3. `bin/rubric_hash.py compute --write` to refresh `meta.sha256`.
4. Update this file's stamp + any changed anchors, then `bin/rubric_hash.py check`
   (it fails CI if the stored hash is stale or this mirror is out of sync).
5. Note in the changelog below why the scoring changed — old-cohort scores stay
   valid under their own hash but are not comparable across the boundary.

### Changelog

- **v1** (`df901308…`) — initial rubric. 10 dimensions (50 C / 50 J), 5 gates, 5 bands.
