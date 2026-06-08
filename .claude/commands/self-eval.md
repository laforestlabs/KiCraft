---
description: Self-eval regression loop — drive every curated example brief (examples.py EXAMPLE_PROMPTS) end to end, auto-answering any clarifying questions, then grade each with the kicraft.eval rubric and compile a saved report.
argument-hint: "[--limit N | --only 1,3,5 | --no-judge | --judge-model M | --build-timeout S] (optional; default: all briefs, full A–F judge)"
---

Run the KiCraft **self-eval** loop: for every brief in `EXAMPLE_PROMPTS`
(`kicraft/server/examples.py`) drive the full pipeline headlessly to a finished
board — auto-answering any parked clarifying question with the model's own first
suggested option — then score the run with the existing `kicraft.eval` rubric
(Class-C metrics + LLM judge → an A–F grade) and compile a cross-brief report.
The mechanical loop lives in `kicraft.eval.self_eval` (`kicraft-eval-batch`); this
command runs it and interprets the results. Extra flags: `$ARGUMENTS`.

> **Cost & time.** This drives real LLM pipelines (BOM part-resolution dominates
> the spend) plus a deterministic place-and-route per board, so the default run
> (9 briefs, judge on) takes **many minutes** and **spends real money** via the
> capped OpenRouter client. The client's spend guard still caps the day. For a
> fast smoke test, pass `--limit 1`; to skip the judge spend, `--no-judge`.

## 1. Launch the batch (background — it outlives a single turn)

A full run exceeds one foreground step, so launch it in the **background** with a
known output dir and tail the log. Resolve the repo venv the same way the other
KiCraft commands do.

```bash
REPO=$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/KiCraft"); PY="$REPO/.venv/bin/python"
OUT="$REPO/logs/self_eval/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT"
echo "OUT=$OUT"
```

Then start the harness **in the background** (set the Bash tool's
`run_in_background: true`), writing both streams to `$OUT/run.log`. Paste the
`OUT` value literally (shell vars do not persist between Bash calls):

```bash
"$PY" -m kicraft.eval.self_eval --out "<OUT>" $ARGUMENTS > "<OUT>/run.log" 2>&1
```

Tell the user it is running, name `<OUT>`, and note you'll report when it
finishes. While it runs you may peek with `tail -n 30 "<OUT>/run.log"` — each
brief prints a `grade=… final=… build=…` line as it completes. The harness
writes `summary.json` + `summary.md` only at the very end, so treat the existence
of `<OUT>/summary.json` as "done."

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

> `/kicraft-investigate <OUT>/run_NN_…` — to root-cause that run (verdict + ERC
> errors at ×100-corrected coords, code-bug vs model-output).

Keep the final message decision-useful: lead with the headline and the spend,
then the table, then the short "needs attention" list with the investigate
pointer. The saved `summary.md` / `summary.json` are the durable record.
