# Electrical-Review Model Bakeoff

**Picking the LLM for KiCraft's Layer-4 fab gate**

`2026-06-19` · 7 models × 3 reasoning arms (off / medium / high) × 27 designs × K=1
· **566/567 reviews** · **$12.53 LLM spend** · graded by Claude Code
· corpus `logs/bakeoff/20260618T200126Z`

---

## Executive summary

We empirically compared seven mid-range LLMs as the reviewer for KiCraft's
electrical-review fab gate — the Layer-4 check that reads a finished design's
netlist/BOM digest and blocks fabrication (`rc7`) if it finds an electrically
**broken-but-DRC-clean** board. Each model reviewed a frozen, hand-labeled corpus
(6 real blocker designs, 17 sound, 4 synthetic) at three reasoning efforts; Claude
Code graders (separate context, matching-only) converted raw block-flags into true
recall against the labeled defects.

**Bottom line:**

- **Recommendation — switch the default reviewer from `deepseek-v4-flash` to
  `minimax/minimax-m3` at medium reasoning effort.** It is the only model that pairs
  perfect recall (100%) with the lowest over-block rate (14% on clean boards, 20% on
  warning boards), at low cost (~$0.012/board).
- **Bigger finding — every model over-blocks.** Even the best false-blocks 1-in-7
  clean boards; the incumbent `flash` false-blocks *half*. A single skeptical LLM
  pass is a poor **hard** fab gate. The higher-value fix is to redesign the gate (cap
  margin/intent findings at WARNING, or require corroboration), not just swap models.
- **The model ensemble cross-checked our ground-truth labels** — it caught a real
  flaw we had missed (run_05) and exposed *shared* hallucinations (run_17) that no
  single model, or even a majority vote, would catch.

---

## Methodology

- **Corpus:** 28 designs from a frozen self-eval snapshot, audited + datasheet-checked
  into **6 in-distribution blockers** (DRC-clean but electrically wrong), **17 sound**
  (8 truly clean, 9 sound-with-warnings), and **4 injected synthetics**. Labels frozen
  + sha256'd; every model fed the *identical* digest string.
- **Reviewer config (production-faithful):** temperature 0, answer budget **24,000
  tokens** (raised from 3,000 — the 2026 reasoning models emit 10–23k reasoning tokens
  and were truncating before they answered). KiCraft's cost-safety provider routing was
  relaxed for the review call so non-flash models route.
- **Metrics:**
  - **Recall** — of the 6 blockers, fraction where the model emitted a `blocker`
    finding that *semantically matched* the labeled defect (a defect graded
    warning/note counts as a **miss** — operationally the board still ships).
  - **FBRc** — false-block rate on the 7 truly-clean boards (pure over-block).
  - **FBRw** — block rate on the 10 warning boards (escalating a warning to a gate-fail).
  - **$/board**, **s/board** — average cost and wall-time per review.

---

## Results — all models × arms

★ marks each model's best arm (max recall − FBRc). Lower FBRc/FBRw is better.

| Model | Arm | Recall | FBRc | FBRw | Synth | JSON-ok | $/board | s/board | p90 (s) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| deepseek-v4-flash *(incumbent)* | off | 83% | 57% | 50% | 50% | 100% | $0.0010 | 41 | 77 |
| | medium | 83% | 71% | 90% | 100% | 100% | $0.0009 | 38 | 61 |
| | ★ high | 83% | 43% | 60% | 50% | 100% | $0.0010 | 35 | 65 |
| deepseek-v4-pro | ★ off | 100% | 57% | 60% | 100% | 100% | $0.0187 | 96 | 135 |
| | medium | 100% | 57% | 60% | 75% | 100% | $0.0158 | 110 | 160 |
| | high | 83% | 57% | 80% | 25% | 100% | $0.0144 | 84 | 117 |
| **minimax-m3** | off | 83% | 14% | 40% | 75% | 96% | $0.0165 | 202 | 329 |
| | **★ medium** | **100%** | **14%** | **20%** | 75% | 100% | **$0.0125** | 159 | 388 |
| | high | 83% | 14% | 30% | 50% | 100% | $0.0138 | 165 | 289 |
| qwen3.7-plus | off | 67% | 43% | 40% | 75% | 100% | $0.0081 | 124 | 196 |
| | medium | 83% | 43% | 70% | 75% | 100% | $0.0087 | 120 | 191 |
| | ★ high | 83% | 29% | 50% | 50% | 100% | $0.0082 | 116 | 185 |
| glm-5.2 | off | 100% | 43% | 60% | 75% | 100% | $0.0321 | 165 | 272 |
| | medium | 100% | 29% | 40% | 75% | 100% | $0.0367 | 156 | 366 |
| | ★ high | 100% | 14% | 40% | 75% | 100% | $0.0361 | 191 | 422 |
| mistral-medium-3-5 | off | 83% | 100% | 100% | 100% | 100% | $0.0107 | 11 | 22 |
| | medium | 83% | 100% | 100% | 100% | 100% | $0.0774 | 82 | 161 |
| | high | 83% | 100% | 100% | 100% | 100% | $0.0704 | 64 | 100 |
| claude-haiku-4.5 *(ref)* | off | 33% | 100% | 100% | 100% | 100% | $0.0095 | 14 | 18 |
| | ★ medium | 67% | 43% | 70% | 100% | 100% | $0.0308 | 46 | 66 |
| | high | 67% | 57% | 70% | 75% | 100% | $0.0407 | 62 | 83 |

---

## Per-model breakdown

### `deepseek/deepseek-v4-flash` — incumbent
By far the **cheapest ($0.001) and fastest (~35s)**. Catches 5 of 6 blockers and
never fails to return JSON. But it **over-blocks 43–71% of clean boards**, and —
unusually — **reasoning makes it worse** (medium FBRc 71% vs off 57%): extra
deliberation makes the small model more trigger-happy. Misses the subtle R-2R
topology defect and flags only half the synthetic floor on its off/high arms.

### `deepseek/deepseek-v4-pro`
Full recall on its off/medium arms, but **over-blocks 57%** of clean boards with no
improvement from reasoning. Reasons heavily (~20k tokens) → slow (~100s) and 10–20×
flash's cost. Good detector, poor precision — not worth the price over minimax.

### `minimax/minimax-m3` — **the winner**
At medium effort it is **the only model with 100% recall AND the lowest over-block
(14% clean / 20% warning — best on both)**, at a modest **$0.012/board**. Reasoning
helps it (off→medium lifts recall 83→100%). Downsides: **slow** (~2.5 min average,
p90 6.5 min) and it can over-reason on complex digests — it stalled once (run_18, off
arm, ~15 min, no answer). At medium it was reliable; the gate is fail-soft, so a stall
degrades to "skipped", not a hung build.

### `qwen/qwen3.7-plus`
The **cheap-and-decent** option ($0.008). High effort gives 83% recall / 29% FBRc —
weaker recall than minimax/glm but the best precision among the truly low-cost models.
A reasonable budget pick if minimax's latency is unacceptable and flash over-blocks
too much.

### `z-ai/glm-5.2` — recommended fallback
The strongest alternative: **100% recall at every arm**, and high effort drops FBRc to
14% (reasoning clearly helps: 43→14% off→high). **More consistent than minimax** (no
stalls). But **3× the cost ($0.036)**, **2× the warning over-block (40% vs 20%)**, and
the slowest p90 (7 min). The fallback if minimax's reliability/latency disqualifies it.

### `mistralai/mistral-medium-3-5`
**Unusable as a gate: blocks 100% of clean boards at every effort** — it would reject
every good design. (It also rejects the production `reasoning={max_tokens}` format with
a 400, accepting only `effort`.) High recall is meaningless without precision.

### `anthropic/claude-haiku-4.5` — reference only
Poor: off-arm catches just 33% of blockers while blocking 100% of clean boards;
reasoning helps (medium 67% / 43%) but it never becomes competitive, at $0.01–0.04/board.

---

## Notable successes & failures

- **Real defect the ensemble caught that we MISLABELED:** run_05 (USB-PD trigger) —
  20/21 cells flagged that the CH224K's open-drain power-good pin drives the indicator
  LED backwards (it can never light). A genuine flaw we missed (warning-severity; we
  reclassified the label post-grading).
- **Shared hallucination that corroboration CANNOT fix:** run_17 (AL8860 LED driver) —
  17/21 cells "found" a 0.2V sense reference setting 2A on a 1.5A part. The part is
  0.1V → 1A (sound, user-confirmed). A majority vote would have wrongly blocked it.
- **Scattered hallucinations (filtered by corroboration):** run_04 crossover (4/21 —
  botched the filter math; 10µF *is* correct for 2kHz), run_01 RC filter (5/21 — a valid
  adjustable topology).
- **Recurring single-defect hallucination:** run_22 (motor board) — many models "found"
  an incomplete feedback divider on the AP63203, which is a **fixed 3.3V part**. Graders
  correctly routed these to "unmatched", so they did not inflate recall.
- **Hardest real blocker:** run_02 (R-2R DAC) — only **12/21** matched the labeled
  topology defect; several models instead flagged a *different* real issue (missing
  ladder termination / op-amp feedback). Recall is sensitive to which of multiple true
  faults a model fixates on.
- **Easiest real blockers:** the concrete structural/value faults — A4988 sense
  resistors (run_27, **19/21**), TPS5430 wrong feedback voltage (run_15, **19/21**),
  no-firmware-path (run_22, **19/21**), relay missing input connector (run_19, **20/21**).

---

## Conclusion & recommendation

**Set `KICRAFT_REVIEW_MODEL = minimax/minimax-m3` with reasoning effort = `medium`.**
Across the corpus it is the single best reviewer: **100% recall** on real blockers, the
**lowest false-block rate of any model** (14% clean / 20% warning), at **~$0.012/board**.
It strictly dominates the incumbent flash, which catches fewer blockers (83%) and
false-blocks far more (43–71% of clean boards). `z-ai/glm-5.2` at high effort is the
recommended fallback (equal recall and clean-FBR, more reliable, but 3× the cost, 2× the
warning over-block, and slower).

**The cost of the switch is latency:** minimax averages ~2.5 min per review and can reach
~6.5 min, vs ~35s for flash. Because the gate runs **once per build**, is **fail-soft**
(any error/timeout degrades to "skipped", never a hung build), and is **opt-out** via
`KICRAFT_ELECTRICAL_REVIEW=0`, this is an acceptable trade for materially better
defect-catching and far fewer wrongly-rejected good designs. The 24k answer budget
(`review_max_tokens`) and review-only routing relaxation (`Settings.for_review()`) are
already in place to support it.

**However — the dominant result is that *every* model over-blocks.** The best still
false-blocks 1-in-7 clean boards and 1-in-5 warning boards; a single skeptical LLM pass
is fundamentally too aggressive for a *hard* fab gate. The highest-value follow-up is to
**redesign the gate, not just the model**: cap margin / intent / sizing findings at
**WARNING** and hard-block only on the unambiguous classes (reversed power, no
programming path, dead short, grossly out-of-spec rail), and/or require **≥2-model
corroboration** before a hard block. Switching to minimax is the immediate win;
redesigning the gate is the structural fix.

---

## Caveats

- **Small N:** 6 blockers + 7 clean → recall and FBRc resolve to about ±1 case (~15%);
  rankings within ~15% are ties (minimax-medium and glm-high are statistically tied on
  recall + clean-FBR).
- **K=1:** one sample per cell, so temperature-0 consistency was not measured. A
  follow-up K≥3 run on the top 2–3 models would quantify flakiness.
- **Single corpus:** one self-eval snapshot, one design model's digest style. Confirm on
  a second snapshot before locking the model long-term.
- **Imperfect hand-labels:** the ensemble caught run_05; treat the absolute FBR numbers
  as indicative and the relative model ranking as robust.

---

*Artifacts: `scripts/bakeoff_*.py`, frozen corpus + `labels.json` + `results.jsonl` +
`gradings.jsonl` + `scorecard.md` + `report_data.json` under
`logs/bakeoff/20260618T200126Z/`.*
