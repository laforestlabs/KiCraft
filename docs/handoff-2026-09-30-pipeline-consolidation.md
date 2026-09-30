# Handoff — pipeline consolidation, deployed 2026-09-30

**Status:** deployed to production `kicraft.io` at 2026-09-30T15:44Z.
**Deployed revision:** `da2e8b2` (17 commits on top of the pre-session tip `b40ee8f`).
**Rollback:** `git -C ~/KiCraft checkout b40ee8f && ./deploy/deploy-production.sh`.
**Health at deploy:** both services restarted (`kicraft-web` pid 1279956, `kicraft-build-worker`
pid 1279987), `curl -sf http://127.0.0.1:8080/` → 200, worker log ends
`[build-worker] ready (max 1 concurrent build(s))`, no tracebacks in `logs/kicraft_web.log`, and
every new gate/module is importable from the deployed package. No dependency changes, so the
`pip install -e` step of the AGENTS.md flow was unnecessary; the deploy was
`./deploy/deploy-production.sh` only.

## 1. The number that matters

Three fresh 34-brief campaigns per arm, same corpus and settings, one campaign at a time on this
host (a global single build slot serialises builds).

| Arm | Fab-ready exports | Rate | Distinct briefs |
|---|---|---|---|
| baseline (frozen safety-prerequisite checkout) | **15 / 102** | 14.7% | 10 |
| candidate (`da2e8b2`) | **26 / 102** | 25.5% | 13 |

Per campaign: candidate 7, 11, 8 vs baseline 6, 5, 4. **+11 deliveries per 102 attempts — the
preregistered minimum improvement, met exactly**, at 1.7x the baseline rate. Gained briefs:
`can-node`, `hex-env-sensor`, `highside-switch-10a`, `rounded-c3-devboard`, `servo-driver-16`. The
two apparent losses (`star-ornament`, `thermocouple-amp`) are flaky on both arms (1/3 baseline,
different failure reasons per candidate attempt) — sampling, not regressions.

**Not met:** distinct briefs +3 against the preregistered +4; the held-out pair delivers 0 boards
(see §4); the 18-of-34 construction-coverage floor has not been re-measured on this revision; and
"fab-ready export" is a fabrication-gate pass, not an independent fulfillment verdict.

## 2. What changed (17 commits, newest first)

`da2e8b2` final measurement record · `9317cc4` unused published pins become no-connects (also
closed a silent defect tying four ADC ALERT pins) · `544ca47` tactile-button resolves to the
compiler's own momentary button · `919ce64` the USB-C breakout keeps the lowerer that builds its
receptacle · `c7a36a7` five fixes: RP2040 BOOTSEL names a reviewed button, `buck-converter`
resolves, terminals compose at any width, class-based button ownership, KiCad pin-markup folding ·
`04e5cb2` ADC demand spellings · `5f0456a` a terminal block counts **contacts**, not parts ·
`b516f0d` obligation-shape refusals stay repairable · `96e9749` lowerer/derivation fixes ·
`0daa69a` ownership refusals stay repairable, drafts cannot invent obligations · `d9f9bc5` gates
9.43/9.44 (I2C address assignments, analog input ranges) · `ed33e40` non-part demanded classes ·
`70ee845` the auditor no longer answers a question the compiler settles · `f425e68`/`4d23ac3`/
`0c75156` reviewed-part realization and 42 coverage aliases · `20f4147`/`7687b66` obligation
ownership cutover, architecture composition gate, shared build-to-design recovery, derived
assurance/consent labels.

## 3. Deterministic evidence (independent of campaign sampling)

- The seven preregistered negative boundaries all produce their expected disposition; **4/7 before**
  the 9.43/9.44 gate work. Bounds and evidence: `logs/self_eval/pipeline_consolidation_20260928/negative_boundaries` notes and the helpers under `/tmp/nb/`.
- Uncovered demanded-class occurrences over 2,228 saved states: **14.2% → ~6%**.
- `complete_unused_published_ports` changes **11 of 270** recorded committed BOMs, all gains, none
  losing a wired pin.
- Refusals that no longer occur at all in the candidate arm: the obligation-copy refusal, the
  deliberation audit's false catalogue findings (58 → 0 per campaign), the family-mismatch refusal.

## 4. Known limitation (must be labelled wherever coverage is claimed)

The two preregistered **held-out** briefs (RP2040/STM32 + two/four ADS1115, nine/sixteen analog
contacts) produced **0 boards in three rounds of three trials**. Each round got structurally
further: derivation refusal → intent schema → part-count semantics → class spelling → dangling
unused pin → architecture commit gate. In the last round the RP2040 attempt cleared intent,
functional spec, architecture and BOM and failed at the **wiring commit** on a *draft-invented*
`QSPI_CS` net holding one switch pin while the recipe's own QSPI chain was intact — a draft defect
the pipeline may not rewrite into correctness. Treat this family as **review-required / not
delivered**; the assurance surface already distinguishes verified, review-required and
capability-limited output.

## 5. Two corrections future sessions must not re-learn

1. **`PYTHONPATH` does not beat `sys.path[0]`.** Baseline campaigns launched from the main checkout
   silently imported the *candidate* (`baseline_safety_r1`/`r2` recorded this checkout's revision).
   Every comparison before `baseline_safety_r3` is withdrawn. Run a baseline arm with `cwd` in the
   baseline checkout and assert `import kicraft` resolves there:
   `logs/self_eval/pipeline_consolidation_20260928/run_baseline_valid.sh`.
2. **Attribution discipline.** Two of my own root-cause claims were wrong and were corrected by
   measurement: the `architecture_power_block_as_sheet` finding was not causal (a `buck-converter`
   class spelling was), and the `model_authored_protected_identity` refusal was a symptom of
   `_adopt_carrier_families` dropping a lowerer. Both lived in code I had touched earlier the same
   night — the pattern to watch is "my repair made a latent harmful rewrite active".

## 6. How to reproduce or extend

```bash
# candidate arm on the current revision (one campaign at a time; builds serialise on one slot)
cd ~/KiCraft && KICRAFT_LLM_MODE=live .venv/bin/python -u -m kicraft.eval.self_eval \
  --contract-version benchmark-original-v1 --parallel 1 --build-slots 1 --out <dir>

# baseline arm — cwd matters (see §5)
cd ~/KiCraft-baseline-consolidation-20260928 && \
  KICRAFT_LLM_MODE=live ~/KiCraft/.venv/bin/python -u -m kicraft.eval.self_eval \
  --contract-version benchmark-original-v1 --parallel 1 --build-slots 1 --out <dir>

# held-out pair: one full attempt per campaign, so three trials = three campaigns
cd ~/KiCraft && KICRAFT_LLM_MODE=live .venv/bin/python -u -m kicraft.eval.self_eval \
  --brief-manifest logs/self_eval/pipeline_consolidation_20260928/held_out_bundle/manifest.json \
  --obligations   logs/self_eval/pipeline_consolidation_20260928/held_out_bundle/obligations.json \
  --campaign-budget-usd 1.2 --parallel 1 --build-slots 1 --out <dir>

# reviewed-reference replay (deterministic, $0): expect 10 errors, identical to the frozen revision
cd ~/KiCraft && .venv/bin/python -c "from kicraft.eval import design_acceptance as da; print(da.verify_reference_replay())"
```

Campaign directories and per-brief events live under
`logs/self_eval/pipeline_consolidation_20260928/`. Revision-to-arm mapping is in the plan's §14
(`docs/plans/pipeline-architecture-consolidation-2026-09-28.md`).

## 7. Next work, in the order I would take it

1. **Held-out family.** The remaining blockers are narrow but the last one is draft quality, not a
   pipeline gap: consider whether the wiring repair round can be given the model's own detected
   orphan as an explicit *edit* instruction with the recipe's internal net names offered as the
   correct target, rather than only prose. No check may be loosened for it.
2. **The intent stage's `relay-driver-ic` → `relay` mis-read** (`reconciliation.resolve_demanded_classes`):
   it poisoned relay-quad's ownership; the gate now waives the impossible charge, but the committed
   fact stays wrong and costs rounds.
3. **The vendored `srd-05vdc-sl-c` relay bundle** carries a stale `content_hash`, so every run logs
   `skipping 1 broken parts: ['vendored:srd-05vdc-sl-c']` and that reviewed relay is unavailable in
   **both** arms. Remedy (its own change, its own measurement):
   `kicraft validate-part kicraft/parts_library/srd-05vdc-sl-c --update-hash`.
4. **Coverage floor measurement** (18 of 34 supported) on the deployed revision, and the
   fulfillment assessment that separates a fabrication pass from an independently verified board.
5. **Deterministic construction coverage** is 4 of the 23 reference designs that can be probed with
   the architecture gate — the honest denominator for "no BOM/wiring authoring calls".

## 8. Operational notes

- Services are detached `setsid nohup` processes managed by `deploy/restart-web.sh` /
  `deploy/restart-build-worker.sh`; the systemd units are inactive. Use the scripts.
- `deploy/deploy-production.sh` restarts both and verifies HTTP 200 + `[build-worker] ready`; the
  real-provider canary (`deploy/verify-design-canary.sh`) is deliberately not part of it.
- A campaign writes to the tree's fingerprint (`HEAD` + the whole uncommitted diff + untracked
  files under `kicraft`/`deploy`, compared at the campaign's start and end), so **do not edit the
  repo while a campaign runs** — it invalidates that campaign's `campaign_valid`.
