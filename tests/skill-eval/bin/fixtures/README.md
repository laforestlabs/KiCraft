# Scorer regression fixtures

Self-contained run-record fixtures for validating `bin/score_run.py`. They are
**inputs only** (no committed `report.json` — that is run output). Re-run the
scorer after any change to `score_run.py` or `rubric.yaml` and confirm the
expected behaviour below still holds.

## `broken_run/`

A hand-built broken run: `synthesis_check.status = failed` (2 failed checks),
an ERC report with **3 errors** + 1 warning, 7 history entries (2 extra → re-commit
thrash), and 4 accumulated permission entries. Stands in for the original
61-error bmp280 run (whose on-disk artifacts have since been re-synthesised clean).

Expected (rubric v1):

```
.venv/bin/python tests/skill-eval/bin/score_run.py score tests/skill-eval/bin/fixtures/broken_run
```

- `pipeline_completion = 3` (synthesised but status=failed)
- `computing_error_cleanliness = 1` (1–10 ERC errors / ≥2 failed checks)
- `convergence_efficiency = 2` (partial — 2 extra history commits, no transcript)
- `latency` unscored (fallback implausible)
- `interaction_friction = 1` (4 excess permission prompts)
- **script gate `erc_errors ≤ 45` fires**
- With ideal Class-J (all 4) the `finalize` total is **weighted 69 → FINAL 45 → grade D**.

## Real archived records (not in this dir)

`tests/manual-runs/{esp32motionsensor,bmp280-reader}/` are clean on disk
(0 ERC errors; both re-synthesised after the router/footprint fixes landed). Use
them as **parser smoke-tests**: the scorer must read them without error and rank
them well above `broken_run`. With ideal Class-J the esp32motionsensor record
finalizes to **weighted ≈ 84 → grade B** (no gates). The discrimination
`broken_run (D) < esp32motionsensor (B)` is the core scorer regression.

These manual-runs are slated for deletion once the harness is trusted; the
synthetic `broken_run` fixture is what keeps the broken-path regression alive
afterwards.
