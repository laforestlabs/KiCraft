#!/usr/bin/env python3
"""Render the electrical-review model bakeoff report to docs/<name>.pdf (fpdf2)."""
from __future__ import annotations

import json
from pathlib import Path

from fpdf import FPDF

BDIR = Path("logs/bakeoff/20260618T200126Z")
DATA = json.loads((BDIR / "report_data.json").read_text())
OUT = Path("docs/electrical_review_model_bakeoff.pdf")

MODELS = ["flash", "v4pro", "minimax", "qwen", "glm", "mistral", "haiku"]
IDS = {"flash": "deepseek/deepseek-v4-flash (incumbent)", "v4pro": "deepseek/deepseek-v4-pro",
       "minimax": "minimax/minimax-m3", "qwen": "qwen/qwen3.7-plus", "glm": "z-ai/glm-5.2",
       "mistral": "mistralai/mistral-medium-3-5", "haiku": "anthropic/claude-haiku-4.5 (ref)"}
ARMS = ["off", "medium", "high"]

NAVY = (23, 42, 71)
GREY = (90, 90, 90)
LROW = (242, 245, 250)


class PDF(FPDF):
    def multi_cell(self, w, h=None, text="", **kw):
        # guard the common fpdf2 trap: a full-width multi_cell (w=0) called while
        # the cursor sits near the right margin -> "not enough horizontal space".
        if w == 0 and self.x > self.w - self.r_margin - 5:
            self.set_x(self.l_margin)
        return super().multi_cell(w, h, text, **kw)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_y(8)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, f"Electrical-review model bakeoff      p.{self.page_no()}", align="R")
        self.set_text_color(0, 0, 0)
        self.set_xy(self.l_margin, 18)


def h1(pdf, t):
    pdf.ln(2); pdf.set_font("Helvetica", "B", 15); pdf.set_text_color(*NAVY)
    pdf.multi_cell(0, 7, t); pdf.set_text_color(0, 0, 0); pdf.ln(1)


def h2(pdf, t):
    pdf.ln(1.5); pdf.set_font("Helvetica", "B", 11.5); pdf.set_text_color(*NAVY)
    pdf.multi_cell(0, 6, t); pdf.set_text_color(0, 0, 0); pdf.ln(0.5)


def body(pdf, t):
    pdf.set_font("Helvetica", "", 10); pdf.multi_cell(0, 5, t); pdf.ln(0.5)


def bullet(pdf, t, ind=4):
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(pdf.l_margin + ind)
    pdf.multi_cell(0, 5, chr(149) + "  " + t)


def best_arm(m):
    # arm maximizing recall - fbrc (the headline tradeoff)
    arms = [a for a in ARMS if a in DATA[m]]
    def sc(a):
        x = DATA[m][a]; r = x["recall"] if x["recall"] is not None else 0
        return r - x["fbrc"]
    return max(arms, key=sc)


def main():
    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()

    # ---- title block ----
    pdf.set_font("Helvetica", "B", 20); pdf.set_text_color(*NAVY)
    pdf.multi_cell(0, 9, "Electrical-Review Model Bakeoff")
    pdf.set_font("Helvetica", "", 12); pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 6, "Picking the LLM for KiCraft's Layer-4 fab gate")
    pdf.set_text_color(0, 0, 0); pdf.ln(2)
    pdf.set_font("Helvetica", "", 9.5); pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 5, "2026-06-19  |  7 models x 3 reasoning arms (off / medium / high) x 27 designs "
                   "x K=1  |  566/567 reviews  |  $12.53 LLM spend  |  graded by Claude Code")
    pdf.set_text_color(0, 0, 0); pdf.ln(2)

    # ---- executive summary ----
    h1(pdf, "Executive summary")
    body(pdf, "We empirically compared seven mid-range LLMs as the reviewer for KiCraft's "
          "electrical-review fab gate -- the Layer-4 check that reads a finished design's "
          "netlist/BOM digest and blocks fabrication (rc7) if it finds an electrically "
          "broken-but-DRC-clean board. Each model reviewed a frozen, hand-labeled corpus "
          "(6 real blocker designs, 17 sound, 4 synthetic) at three reasoning efforts; "
          "Claude Code graders (separate context, matching-only) converted raw block flags "
          "into true recall against the labeled defects.")
    h2(pdf, "Bottom line")
    bullet(pdf, "RECOMMENDATION: switch the default reviewer from deepseek-v4-flash to "
           "minimax/minimax-m3 at medium reasoning effort. It is the only model that pairs "
           "perfect recall (100%) with the lowest over-block rate (14% on clean boards, 20% "
           "on warning boards) at low cost (~$0.012/board).")
    bullet(pdf, "BIGGER FINDING: every model over-blocks. Even the best false-blocks 1-in-7 "
           "clean boards; the incumbent flash false-blocks half. A single skeptical LLM pass "
           "is a poor HARD fab gate -- the higher-value fix is to redesign the gate (cap "
           "margin/intent findings at WARNING, or require corroboration), not just swap models.")
    bullet(pdf, "The model ensemble cross-checked our ground-truth labels: it caught a real "
           "flaw we had missed (run_05) and exposed shared hallucinations (run_17) that no "
           "single model -- or majority vote -- would catch.")

    # ---- methodology ----
    h1(pdf, "Methodology")
    bullet(pdf, "Corpus: 28 designs from a frozen self-eval snapshot, audited + datasheet-checked "
           "into 6 in-distribution blockers (DRC-clean but electrically wrong), 17 sound (8 truly "
           "clean, 9 sound-with-warnings), and 4 injected synthetics. Labels frozen + sha256'd; "
           "every model fed the IDENTICAL digest string.")
    bullet(pdf, "Reviewer config production-faithful: temperature 0, answer budget 24000 tokens "
           "(raised from 3000 -- the 2026 reasoning models emit 10-23k reasoning tokens and "
           "truncated before answering). KiCraft's cost-safety provider routing was relaxed for "
           "the review call so non-flash models route.")
    bullet(pdf, "Metrics: RECALL = of 6 blockers, fraction where the model emitted a blocker "
           "finding that semantically MATCHED the labeled defect (a defect graded warning/note "
           "= miss). FBRc = false-block rate on the 7 truly-clean boards. FBRw = block rate on "
           "the 10 warning boards. Cost/board and time/board are per-review averages.")

    # ---- results table ----
    h1(pdf, "Results -- all models x arms")
    _table(pdf)
    body(pdf, "Rec = recall on 6 blockers; FBRc/FBRw = false-block on clean / warning boards "
          "(lower is better); $/bd and s/bd = average cost and wall-time per review. Best arm "
          "per model is shaded.")

    # ---- per-model breakdown ----
    pdf.add_page()
    h1(pdf, "Per-model breakdown")
    for m in MODELS:
        ba = best_arm(m); x = DATA[m][ba]
        h2(pdf, f"{IDS[m]}")
        rec = f"{x['recall']*100:.0f}%" if x["recall"] is not None else "-"
        pdf.set_font("Helvetica", "I", 9); pdf.set_text_color(*GREY)
        pdf.multi_cell(0, 4.6, f"best arm: {ba}   recall {rec}   FBRc {x['fbrc']*100:.0f}%   "
                       f"FBRw {x['fbrw']*100:.0f}%   ${x['cost']:.4f}/board   {x['lat']:.0f}s/board "
                       f"(p90 {x['p90']:.0f}s)")
        pdf.set_text_color(0, 0, 0)
        body(pdf, VERDICTS[m])

    # ---- notable ----
    h1(pdf, "Notable successes & failures")
    for t in NOTABLE:
        bullet(pdf, t)

    # ---- conclusion ----
    pdf.add_page()
    h1(pdf, "Conclusion & recommendation")
    for p in CONCLUSION:
        body(pdf, p)
    h2(pdf, "Caveats")
    for t in CAVEATS:
        bullet(pdf, t)

    OUT.parent.mkdir(exist_ok=True)
    pdf.output(str(OUT))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


def _table(pdf):
    cols = [("Model", 30), ("Arm", 15), ("Rec", 13), ("FBRc", 14), ("FBRw", 14),
            ("Synth", 14), ("$/bd", 18), ("s/bd", 14), ("p90", 12)]
    pdf.set_font("Helvetica", "B", 8.5); pdf.set_fill_color(*NAVY); pdf.set_text_color(255, 255, 255)
    for name, w in cols:
        pdf.cell(w, 6.5, name, border=0, align="C", fill=True)
    pdf.ln(); pdf.set_text_color(0, 0, 0)
    for m in MODELS:
        ba = best_arm(m)
        for a in ARMS:
            if a not in DATA[m]:
                continue
            x = DATA[m][a]
            fill = a == ba
            if fill:
                pdf.set_fill_color(*LROW); pdf.set_font("Helvetica", "B", 8.5)
            else:
                pdf.set_font("Helvetica", "", 8.5)
            rec = f"{x['recall']*100:.0f}%" if x["recall"] is not None else "-"
            vals = [IDS[m].split(" ")[0].split("/")[-1], a, rec, f"{x['fbrc']*100:.0f}%",
                    f"{x['fbrw']*100:.0f}%", f"{x['synth']*100:.0f}%", f"${x['cost']:.4f}",
                    f"{x['lat']:.0f}", f"{x['p90']:.0f}"]
            for (name, w), v in zip(cols, vals):
                pdf.cell(w, 5.6, v, border="B", align="C", fill=fill)
            pdf.ln()


VERDICTS = {
 "flash": "The incumbent: by far the cheapest ($0.001) and fastest (~35s). Catches 5 of 6 "
          "blockers and never fails to return JSON. But it over-blocks 43-71% of clean boards, "
          "and -- unusually -- reasoning makes it WORSE (medium-effort FBRc 71% vs off 57%): "
          "extra deliberation makes the small model more trigger-happy. Misses the subtle R-2R "
          "topology defect and only flags half the synthetic floor on its off/high arms.",
 "v4pro": "Full recall on its off/medium arms, but over-blocks 57% of clean boards with no "
          "improvement from reasoning. Reasons heavily (~20k tokens), so it is slow (~100s) and "
          "10-20x flash's cost. Good detector, poor precision -- not worth the price over minimax.",
 "minimax": "The winner. At medium effort it is the only model with 100% recall AND the lowest "
            "over-block (14% clean / 20% warning -- best on both), at a modest $0.012/board. "
            "Reasoning helps it (off->medium lifts recall 83->100%). Downsides: slow "
            "(~2.5 min average, p90 6.5 min) and it can over-reason on complex digests -- it "
            "stalled once (run_18, off arm, ~15 min, no answer). At medium it was reliable; the "
            "gate is fail-soft so a stall degrades to 'skipped', not a hung build.",
 "qwen": "The cheap-and-decent option ($0.008). High-effort gives 83% recall / 29% FBRc -- "
         "weaker recall than minimax/glm but the best precision among the truly low-cost models. "
         "A reasonable budget pick if minimax's latency is unacceptable and flash over-blocks too "
         "much.",
 "glm": "The strongest alternative to minimax: 100% recall at every arm, and high effort drops "
        "FBRc to 14% (reasoning clearly helps it: 43->14% off->high). More consistent than minimax "
        "(no stalls). But 3x the cost ($0.036) and 2x the warning-board over-block (40% vs 20%), "
        "and the slowest p90 (7 min). The recommended fallback if minimax's reliability/latency "
        "disqualifies it.",
 "mistral": "Unusable as a gate: it blocks 100% of clean boards at every effort -- it would reject "
            "every good design. (It also rejects the production reasoning={max_tokens} format with a "
            "400, only accepting effort.) High recall is meaningless without precision.",
 "haiku": "Reference only (not production-eligible). Poor: off-arm catches just 33% of blockers "
          "while blocking 100% of clean boards; reasoning helps (medium 67% recall / 43% FBRc) but "
          "it never becomes competitive, at $0.01-0.04/board.",
}

NOTABLE = [
 "Real defect the ensemble caught that we MISLABELED: run_05 (USB-PD trigger) -- 20/21 cells "
 "flagged that the CH224K's open-drain power-good pin drives the indicator LED backwards (it can "
 "never light). A genuine flaw we missed (warning-severity; we reclassified the label).",
 "Shared hallucination that corroboration CANNOT fix: run_17 (AL8860 LED driver) -- 17/21 cells "
 "'found' a 0.2V sense reference setting 2A on a 1.5A part. The part is 0.1V -> 1A (sound, "
 "user-confirmed). A majority vote would have wrongly blocked it.",
 "Scattered hallucinations (filtered by corroboration): run_04 crossover (4/21, botched the "
 "filter math -- 10uF IS correct for 2kHz), run_01 RC filter (5/21, a valid adjustable topology).",
 "Recurring single-defect hallucination: run_22 (motor board) -- many models 'found' an incomplete "
 "feedback divider on the AP63203, which is a FIXED 3.3V part. Graders correctly routed these to "
 "'unmatched', so they did not inflate recall.",
 "Hardest real blocker: run_02 (R-2R DAC) -- only 12/21 matched the labeled topology defect; "
 "several models instead flagged a different real issue (missing ladder termination / op-amp "
 "feedback), showing recall is sensitive to which of multiple true faults a model fixates on.",
 "Easiest real blockers: the structural/value faults -- A4988 sense resistors (run_27, 19/21), "
 "TPS5430 wrong feedback voltage (run_15, 19/21), no-firmware-path (run_22, 19/21), relay missing "
 "input connector (run_19, 20/21). Models reliably catch concrete, citable defects.",
]

CONCLUSION = [
 "Set KICRAFT_REVIEW_MODEL = minimax/minimax-m3 with reasoning effort = medium. Across the "
 "corpus it is the single best reviewer: 100% recall on real blockers, the lowest false-block "
 "rate of any model (14% clean / 20% warning), at ~$0.012 per board. It strictly dominates the "
 "incumbent flash, which catches fewer blockers (83%) and false-blocks far more (43-71% of clean "
 "boards). z-ai/glm-5.2 at high effort is the recommended fallback (equal recall and clean-FBR, "
 "more reliable, but 3x the cost, 2x the warning over-block, and slower).",
 "The cost of the switch is latency: minimax averages ~2.5 minutes per review and can reach ~6.5 "
 "minutes, versus ~35 seconds for flash. Because the gate runs once per build, is fail-soft (any "
 "error/timeout degrades to 'skipped', never a hung build), and is opt-out via "
 "KICRAFT_ELECTRICAL_REVIEW=0, this is an acceptable trade for materially better defect-catching "
 "and far fewer wrongly-rejected good designs. The answer budget (review_max_tokens=24000) and "
 "review-only routing relaxation are already in place to support it.",
 "However -- the dominant result is that EVERY model over-blocks. The best still false-blocks "
 "1-in-7 clean boards and 1-in-5 warning boards; a single skeptical LLM pass is fundamentally too "
 "aggressive for a HARD fab gate. The highest-value follow-up is therefore to redesign the gate, "
 "not just the model: cap margin / intent / sizing findings at WARNING and hard-block only on the "
 "unambiguous classes (reversed power, no programming path, dead short, grossly out-of-spec rail), "
 "and/or require >=2-model corroboration before a hard block. Switching to minimax is the "
 "immediate win; redesigning the gate is the structural fix.",
]

CAVEATS = [
 "Small N: 6 blockers + 7 clean -> recall and FBRc resolve to about +/-1 case (~15%); rankings "
 "within ~15% are ties (minimax-medium and glm-high are statistically tied on recall+clean-FBR).",
 "K=1: one sample per cell, so temperature-0 consistency was not measured. A follow-up K>=3 run "
 "on the top 2-3 models would quantify flakiness.",
 "Single corpus: one self-eval snapshot, one design model's digest style. Confirm on a second "
 "snapshot before locking the model long-term.",
 "Our hand-labels were imperfect (the ensemble caught run_05); treat the absolute FBR numbers as "
 "indicative, the relative model ranking as robust.",
]

if __name__ == "__main__":
    main()
