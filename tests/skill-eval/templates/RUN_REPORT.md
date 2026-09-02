<!--
RUN_REPORT template. Copy into the run record (~/kicraft-eval/runs/<run-id>/RUN_REPORT.md)
and fill it in. `report.json` (written by score_run.py) is the machine sidecar; this
markdown is the human narrative. Keep the two consistent — the scorecard here must
match report.json's dimensions/score.
Delete these comments when filling in.
-->
# KiCraft run report — <scenario id>: <title>

| field | value |
|---|---|
| Run ID | `<run-id>` |
| Scenario | `<S0x>` — <one-line> |
| Date | <UTC> |
| **Rubric** | **v<N> · `sha256:<full hash>`** |
| Target mode | release / dev |
| Skill provenance | `skill_sha256:<…>` (from run.json) |
| CLI | `<path>` |
| Subject session | `<agent runtime + session id>` |
| Transcript | `<run-dir>/transcript.jsonl` |

> Scores are only comparable to other runs under the **same rubric sha256**.

## Executive summary

<3–6 sentences: what was designed, did it complete, the headline finding(s), and
the verdict. State the final score and grade up front.>

## Scorecard

<Paste from `score_run.py finalize`. The numbers MUST equal report.json.>

| Dimension | Class | Wt | Lvl | Pts | Note |
|---|---|---|---|---|---|
| pipeline_completion | C | 12 | _ | _ | |
| computing_error_cleanliness | C | 16 | _ | _ | |
| convergence_efficiency | C | 8 | _ | _ | |
| latency | C | 6 | _ | _ | |
| interaction_friction | C | 8 | _ | _ | |
| spec_compliance | J | 10 | _ | _ | |
| intent_fidelity | J | 10 | _ | _ | |
| electrical_soundness | J | 16 | _ | _ | |
| part_selection_quality | J | 8 | _ | _ | |
| failure_honesty | J | 6 | _ | _ | |

- **Weighted:** _ / 100
- **Gates triggered:** <none | id ≤ cap (why)>
- **FINAL:** _ → grade **_** → **<VERDICT>**

## Deterministic metrics

<Paste the metrics block from `score_run.py score`. The objective baseline.>

```
synthesized        : …
ERC                : … errors / … warnings
failed checks      : …
latency            : … min
user questions     : …   (band …)
failed_commits     : …   crashes : …
permission floor   : …   (excess …)
token usage        : … tok over … call(s)  est ~$…
```

## Findings

> Severity: **P0** ship-blocker · **P1** real bug / spec violation · **P2** UX/quality · **P3** minor.
> Class: **C** computing (machine-detectable) · **J** judgment (design/strategy/intent).

| ID | Class | Sev | Area | Status | Summary |
|----|-------|-----|------|--------|---------|
| F1 | C/J | P? | <stage / synthesis / install / skill> | open/fixed/self-healed | <one line> |

(Expand each material finding below with evidence: transcript line / artifact / quote.)

### F1 — <title> (<Class> · <Sev>)
**Symptom.** <what was observed, with evidence>
**Root cause / where.** <file:line or stage spec>
**Why it matters.** <impact on the delivered board or the user>

## Timeline

| time | event | notes |
|------|-------|-------|
| | session start | |
| | intent committed | |
| | … | |
| | synthesized | status, ERC |

## Per-stage grading (vs the stage spec)

For each of intent / functional_spec / architecture / bom / wiring / synthesis:
**PASS / FAIL** + one paragraph judged against
`.agents/skills/kicraft/stages/<stage>.md`.

- **intent** — PASS/FAIL: …
- **functional_spec** — …
- **architecture** — …
- **bom** — …
- **wiring** — …
- **synthesis** — …

## Electrical-design review (the gotcha layer)

This drives the `electrical_soundness` dimension and the
`unprogrammable_mcu` / `silent_substitution` gates. Walk the checklist; cite
the BOM/wiring/architecture slot as evidence.

- **Grounding / ground loops:** …
- **Decoupling / bypass:** …
- **MCU programming / first-flash path:** … (present? surfaced if absent?)
- **Input / ESD protection:** …
- **Strap / pull resistors (boot/EN/reset):** …
- **Regulator thermal headroom & rail current sizing:** …
- **Polarity / orientation (diodes, polarized caps, connectors):** …
- **Analog/digital separation (mixed-signal only):** …

## Fix-plan for the implementation agent

Tiered, pick-up-cold. Each item: **Symptom / Where / Concretely / Acceptance**.
Order P0 → P3; within a tier, items are independent unless noted.

### P0 — ship-blocking
#### <id> <title>
- **Symptom.** …
- **Where.** `<path>` (or stage spec / install path)
- **Concretely.** <numbered steps>
- **Acceptance.** <how the implementer knows it's fixed; ideally a test>

### P1 — high
### P2 — medium
### P3 — minor / upstream

## Notes for the next eval run

<Anything that would make the next run of this scenario cleaner: allowlist gaps,
scenario tweaks, signals that were hard to capture.>
