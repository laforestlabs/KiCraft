#!/usr/bin/env python3
"""Electrical-review model bakeoff — matrix runner.

Feeds every model the IDENTICAL frozen digest string for each corpus design and
records its findings, cost, latency and reasoning behaviour. The per-cell review
loop is a VERBATIM replica of electrical_review.review_design (electrical_review.py
:246-268) — same messages, same validation, same retry/correction text — so model
behaviour is identical to production. The only differences are instrumentation and
a richer meta_ctx so the spend ledger attributes cost per (model, cell). Keep this
loop in sync with review_design if that function changes.

Matrix (cost-aware tiers; all slate models support reasoning per the live catalog):
  incumbent/cheap : both arms (off, reasoning=8000), K=3 reps
  pricey          : off K=2, reasoning=8000 K=1
  reference       : both arms K=1 (NOT production-eligible)

Usage:
  python scripts/bakeoff_review_models.py --dry-run        # plan + cost estimate
  python scripts/bakeoff_review_models.py --smoke          # flash x 2 designs x 1
  python scripts/bakeoff_review_models.py                  # full matrix (resumable)
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kicraft.design.synthesis.electrical_review import (
    _build_messages,
    _extract_json,
    _validate,
    has_blocker,
)
from kicraft.server.client import make_client
from kicraft.server.config import Settings

BAKEOFF_DIR = Path("logs/bakeoff/20260618T200126Z")

# Resolved against the live OpenRouter catalog 2026-06-18 (id, $in/$out per Mtok).
SLATE = [
    {"id": "deepseek/deepseek-v4-flash",  "label": "flash",   "tier": "incumbent", "price": (0.090, 0.180)},
    {"id": "deepseek/deepseek-v4-pro",    "label": "v4pro",   "tier": "cheap",     "price": (0.435, 0.870)},
    {"id": "minimax/minimax-m3",          "label": "minimax", "tier": "cheap",     "price": (0.300, 1.200)},
    {"id": "qwen/qwen3.7-plus",           "label": "qwen",    "tier": "cheap",     "price": (0.320, 1.280)},
    {"id": "z-ai/glm-5.2",                "label": "glm",     "tier": "pricey",    "price": (1.200, 4.200)},
    # google/gemini-3.5-flash DROPPED (user OK): truncates at finish=length and
    # is the priciest tier ($9/Mtok out).
    {"id": "mistralai/mistral-medium-3-5","label": "mistral", "tier": "pricey",    "price": (1.500, 7.500)},
    {"id": "anthropic/claude-haiku-4.5",  "label": "haiku",   "tier": "ref",       "price": (1.000, 5.000)},
]

# Reasoning-ON arm uses an explicit OpenRouter effort level (--effort), not a raw
# token budget -- more comparable across the heterogeneous slate (OpenRouter
# normalises effort<->thinking-budget per provider). Default "high" = the
# strongest standard tier (one below the absolute max where a provider exposes
# one). The arm is labelled by the effort string in results.jsonl (e.g. "high").
DEFAULT_EFFORTS = "medium,high"


def make_arms(efforts_csv: str):
    arms = [("off", None)]
    for e in [x.strip() for x in efforts_csv.split(",") if x.strip()]:
        arms.append((e, {"effort": e}))
    return arms


def bakeoff_settings():
    """Settings with KiCraft's production cost-safety routing RELAXED so the
    pricier/closed-weight slate is reachable. Production make_client pins
    provider.order to fp8 backends and caps provider.max_price at $0.18/$0.35,
    which 404s every model but flash. We drop both here; the spend-guard daily/
    total $ ceiling still bounds total spend. NOTE for adoption: the live review
    gate inherits this same cap+pin, so adopting any non-flash reviewer also
    needs the production routing loosened (see scorecard)."""
    s = Settings.from_env()
    s.provider_order = []        # drop the fp8 provider pin (closed-weight need this)
    s.max_price_prompt = 0.0     # drop the prompt price cap
    s.max_price_completion = 0.0  # drop the completion price cap
    return s


def reps_for(tier: str, arm: str) -> int:
    if tier in ("incumbent", "cheap"):
        return 3
    if tier == "pricey":
        return 2 if arm == "off" else 1
    return 1  # ref: both arms, K=1


def scored_design_ids(labels: dict) -> list[str]:
    ids = [d["design_id"] for d in labels["designs"] if d["role"] in ("blocker", "sound")]
    ids += [s["design_id"] for s in labels["synthetics"]]
    return ids


def review_cell(client, digest, model, arm, reasoning, cell_id,
                max_tokens=24000, temperature=0.0, max_attempts=2) -> dict:
    """Replica of review_design's loop, instrumented + custom meta_ctx. NOTE:
    max_tokens is raised from production's 3000 to 24000 -- the 2026 reasoning
    models emit 10-23k reasoning tokens by default and truncate (finish=length)
    at 3000 before writing the JSON answer. Cheap models stop naturally well
    under 24000, so this only costs the heavy reasoners (which is their real
    cost). Adopting any such model in production would likewise need a raised
    review answer budget."""
    messages = _build_messages(digest)
    total_cost = 0.0
    last_text = ""
    error = None
    reasoning_present = False
    reasoning_len = 0
    finish = None
    t0 = time.monotonic()
    used = max_attempts
    ok = False
    findings: list = []
    for attempt in range(max_attempts):
        res = client.chat(
            messages, model=model, max_tokens=max_tokens, temperature=temperature,
            reasoning=reasoning,
            meta_ctx={"phase": "bakeoff", "model_tested": model, "arm": arm,
                      "cell": cell_id, "attempt": attempt})
        last_text = res.get("text") or ""
        total_cost += float(res.get("cost_usd") or 0.0)
        r = res.get("reasoning")
        if r:
            reasoning_present = True
            reasoning_len = max(reasoning_len, len(r))
        finish = res.get("finish_reason")
        ok, findings, error = _validate(_extract_json(last_text))
        if ok:
            used = attempt + 1
            break
        messages.append({"role": "assistant", "content": last_text})
        messages.append({"role": "user", "content":
                         f"That response was not acceptable: {error}. Return ONLY the JSON "
                         "object with a 'findings' array; each finding needs a severity "
                         "(blocker|warning|note), area, issue, and suggestion."})
    latency = round(time.monotonic() - t0, 2)
    if not ok:
        findings = []
    sev = [f.get("severity") for f in findings]
    return {
        "cell": cell_id, "model": model, "arm": arm,
        "ok": ok, "error": None if ok else (error or "no valid verdict"),
        "findings": findings, "has_blocker": has_blocker(findings),
        "n_blocker": sev.count("blocker"), "n_warning": sev.count("warning"),
        "n_note": sev.count("note"),
        "cost_usd": round(total_cost, 6), "latency_s": latency,
        "reasoning_present": reasoning_present, "reasoning_len": reasoning_len,
        "finish_reason": finish, "attempts_used": used,
    }


def build_cells(designs, models, smoke, arms, reps_override=None):
    if smoke:
        models = [m for m in models if m["label"] == "flash"]
        designs = designs[:2]
    cells = []
    for m in models:
        for arm_name, payload in arms:
            if smoke:
                k = 1
            elif reps_override:
                k = reps_override
            else:
                k = reps_for(m["tier"], arm_name)
            for rep in range(k):
                for did in designs:
                    cells.append((m, arm_name, payload, rep, did))
    return cells


def est_cost(cells, digests):
    # ballpark only; smoke + ledger give truth. assume input=digest/4+800,
    # output=600 (+4000 reasoning tokens on the r8000 arm, ~half the budget used)
    total = 0.0
    for (m, arm, _r, _rep, did) in cells:
        intok = len(digests[did]) / 4 + 800
        outtok = 600 + (4000 if arm != "off" else 0)
        pin, pout = m["price"]
        total += intok * pin / 1e6 + outtok * pout / 1e6
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bakeoff-dir", default=str(BAKEOFF_DIR))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", help="comma-separated labels subset")
    ap.add_argument("--efforts", default=DEFAULT_EFFORTS,
                    help="comma list of reasoning efforts for the ON arms, e.g. medium,high")
    ap.add_argument("--reps", type=int, default=None, help="override reps (K) for every cell")
    ap.add_argument("--limit-designs", type=int, default=None,
                    help="use only the first N scored designs (pilot)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    arms = make_arms(args.efforts)

    bdir = Path(args.bakeoff_dir)
    labels = json.loads((bdir / "labels.json").read_text())
    design_ids = scored_design_ids(labels)
    if args.limit_designs:
        design_ids = design_ids[:args.limit_designs]
    digests = {d: (bdir / "corpus" / d / "digest.txt").read_text() for d in design_ids}

    models = SLATE
    if args.models:
        want = set(args.models.split(","))
        models = [m for m in SLATE if m["label"] in want]

    cells = build_cells(design_ids, models, args.smoke, arms, args.reps)
    out = bdir / ("results_smoke.jsonl" if args.smoke else "results.jsonl")

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["cell"])
    todo = []
    for (m, arm, reasoning, rep, did) in cells:
        cid = f"{m['label']}|{did}|{arm}|r{rep}"
        if cid not in done:
            todo.append((m, arm, reasoning, rep, did, cid))

    print(f"slate: {[m['label'] for m in models]}")
    print(f"designs: {len(design_ids)} | total cells: {len(cells)} | "
          f"done: {len(done)} | todo: {len(todo)}")
    print(f"est. cost (ballpark) for todo: ${est_cost([(m,a,r,rp,d) for (m,a,r,rp,d,_c) in todo], digests):.2f}")
    print(f"output -> {out}")
    if args.dry_run:
        by_model = {}
        for (m, a, _r, _rp, _d, _c) in todo:
            by_model[m["label"]] = by_model.get(m["label"], 0) + 1
        print("cells/model:", by_model)
        return

    settings = bakeoff_settings()
    write_lock = threading.Lock()
    fh = out.open("a")

    def worker(m, arm, reasoning, rep, did, cid):
        client = make_client(settings)  # one client per worker (WAL sqlite ledger)
        try:
            rec = review_cell(client, digests[did], m["id"], arm, reasoning, cid)
        except Exception as e:  # a hard failure (budget/API) is recorded, not fatal
            rec = {"cell": cid, "model": m["id"], "arm": arm, "ok": False,
                   "error": f"{type(e).__name__}: {e}", "findings": [],
                   "has_blocker": False, "cost_usd": 0.0, "exception": True}
        with write_lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        return rec

    n = 0
    spent = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, m, arm, reasoning, rep, did, cid)
                for (m, arm, reasoning, rep, did, cid) in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            n += 1
            spent += float(rec.get("cost_usd") or 0.0)
            flag = "" if rec.get("ok") else f"  !{rec.get('error','')[:40]}"
            print(f"[{n}/{len(todo)}] {rec['cell']:42s} "
                  f"blk={int(rec.get('has_blocker', False))} "
                  f"${rec.get('cost_usd', 0):.4f} {rec.get('latency_s', '?')}s{flag}", flush=True)
    fh.close()
    print(f"\ndone: {n} cells, ~${spent:.3f} this run -> {out}")


if __name__ == "__main__":
    main()
