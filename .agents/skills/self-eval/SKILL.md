---
name: self-eval
description: Run KiCraft's curated end-to-end self-evaluation batch, monitor it, and interpret the saved scorecard. Use for regression sweeps, selected example briefs, resumed evaluations, or requests to compare fab readiness and rubric grades.
compatibility: Requires the KiCraft server and evaluation dependencies, configured provider credentials, routing tools, and permission to spend provider budget.
---

Run the KiCraft **self-eval** loop: for every brief in `EXAMPLE_PROMPTS`
(`kicraft/server/examples.py`) drive the full pipeline headlessly to a finished
board — auto-answering any parked clarifying question with the model's own first
suggested option — then score the run with the existing `kicraft.eval` rubric
(Class-C metrics + LLM judge → an A–F grade) and compile a cross-brief report.
The mechanical loop lives in `kicraft.eval.self_eval` (`kicraft-eval-batch`); this
skill runs it and interprets the results. Derive `<USER_FLAGS>` from the user's request, using no extra flags when none were supplied.

> **Cost & time.** This drives real LLM pipelines (BOM part-resolution dominates
> the spend) plus a deterministic place-and-route per board, so the default run
> (9 briefs, judge on) takes **~1 hour** and **spends real money** via the
> capped OpenRouter client. The client's spend guard still caps the day. For a
> fast smoke test, pass `--limit 1`; to skip the judge spend, `--no-judge`.
>
> The harness itself defaults to `--parallel 3 --build-slots 2` (briefs overlap;
> at most 2 routing JVMs at once) — no flags needed. Pass `--parallel 1` only
> when a strictly sequential baseline run is explicitly wanted.

## 1. Launch the batch (background — it outlives a single turn)

A full run exceeds one foreground step, so launch it in the **background** with a
known output dir and tail the log. Resolve the repo venv the same way the other
KiCraft commands do.

```bash
REPO=$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/KiCraft"); PY="$REPO/.venv/bin/python"
OUT="$REPO/logs/self_eval/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT"
echo "OUT=$OUT"
```

Then start the harness using the active agent runtime's supervised background-process facility, writing both streams to `<OUT>/run.log`. Paste the resolved output path literally because shell variables may not persist between command calls:

```bash
"$PY" -m kicraft.eval.self_eval --out "<OUT>" <USER_FLAGS> > "<OUT>/run.log" 2>&1
```

Tell the user it is running, name `<OUT>`, and note you'll report when it
finishes. While it runs you may peek with `tail -n 30 "<OUT>/run.log"` — each
brief prints a `grade=… final=… build=…` line as it completes. The harness
checkpoints `summary.json` after **every** brief (live progress, and what
`--resume <OUT>` reads to finish an interrupted batch), so its existence does
NOT mean "done" — the run is finished only when `summary.json` has top-level
`finished_at`/`wall_s` keys (written at the very end) or the background process
has exited.

## 2. Present the compiled report

Once the background run exits (or `<OUT>/summary.json` exists), read it and give
the user a crisp scorecard — do **not** just dump the JSON:

```bash
"$PY" - "<OUT>/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"briefs={s['n']} graded={s['graded_n']} fab_ready={s['fab_ready']}/{s['n']} "
      f"errored={s['n_errored']} mean={s['mean_final']} median={s['median_final']}")
print("grades:", s["grade_counts"], " gates:", s["gate_counts"], f" spend=${s['total_cost_usd']}")
print(f"design_model={s.get('design_model')}  judge={s.get('judge_model') or 'off'}")
for r in s["runs"]:
    tag = r.get("grade") or ("ERR" if r.get("error") else r.get("design_status","?"))
    print(f"  #{r['index']:>2} {tag:>4} final={r.get('final')} "
          f"build={r.get('build_label')} Q={r.get('questions')} "
          f"${(r.get('design_cost_usd') or 0)+(r.get('judge_cost_usd') or 0):.4f}  {r['prompt'][:54]}")
    if r.get("error"): print(f"        ERROR: {r['error']}")
PY
```

Summarise for the user:
- the **headline**: how many briefs reached a fab-ready board, the grade
  distribution, mean/median score, and total spend;
- a compact per-brief table (grade · final · verdict · build label · cost · brief),
  reusing `summary.md` which already renders one;
- **regressions / things to fix**, called out explicitly: any errored brief, any
  triggered gate (e.g. `erc_errors` caps the grade at 45, `synthesis_broken`,
  `unprogrammable_mcu`), any `final < 60` (REWORK/NOT-READY/BROKEN), and any build
  that is not `fab-ready` (e.g. `ERC errors`, `route/infra failed`).

## 3. Point at the next step for failures

For each errored / low-grade / not-fab-ready brief, the run dir
(`<OUT>/run_NN_…/`) holds the full evidence: `eval/report.json` (per-dimension
levels + judge rationale), `events.jsonl`, `.kicraft/state.json`, and the
generated KiCad tree with the ERC report. Hand the user a ready next move:

> Activate the `kicraft-investigate` skill on `<OUT>/run_NN_…` to root-cause that run (verdict + ERC errors at ×100-corrected coordinates, code bug vs model output).

Keep the final message decision-useful: lead with the headline and the spend,
then the table, then the short "needs attention" list with the investigate
pointer. The saved `summary.md` / `summary.json` are the durable record.
