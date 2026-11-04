#!/usr/bin/env python3
"""Drive fresh "Surprise me" briefs through the real design+build pipeline.

Mirrors the web worker / self-eval path: the five LLM design stages
(``run_design``) then the deterministic synth+place+route+fab build
(``run_build``), for a set of seeds drawn from ``generate_brief``. Used to
measure the routing tail on boards nobody hand-picked -- the fix being judged is
in the build tail, so the brief set must not be the frozen corpus.

Usage:
    scripts/routing_surprise_cohort.py --count 5
    scripts/routing_surprise_cohort.py --seeds 17,42,99

Writes ``<out>/summary.json`` plus per-brief ``<out>/<seed>-<slug>/`` run dirs
and prints a table. Costs real provider spend (per-project cap applies).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from kicraft.eval.self_eval import run_build, run_design  # noqa: E402
from kicraft.server.examples import BRIEF_SEED_MODULO, generate_brief  # noqa: E402


def _slug(text: str, limit: int = 40) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(words[:6])[:limit] or "brief"


def _routing_summary(rundir: Path) -> dict:
    """Leaf/parent routing outcome from the autoexperiment's own records."""
    out: dict = {}
    gen = rundir / "generated"
    stems = [p for p in gen.iterdir() if p.is_dir()] if gen.is_dir() else []
    for stem in stems:
        exp = stem / ".experiments"
        if not exp.is_dir():
            continue
        out["stem"] = stem.name
        rounds: list[dict] = []
        log = exp / "experiments.jsonl"
        if log.is_file():
            for line in log.read_text(encoding="utf-8").splitlines():
                try:
                    rounds.append(json.loads(line))
                except ValueError:
                    continue
        out["rounds"] = [
            {
                "round_num": r.get("round_num"),
                "keep": r.get("kept"),
                "leaf_total": r.get("leaf_total"),
                "leaf_accepted": r.get("leaf_accepted"),
                "parent_composed": r.get("parent_composed"),
                "parent_routed": r.get("parent_routed"),
                "score": r.get("score"),
            }
            for r in rounds
        ]
        status = exp / "run_status.json"
        if status.is_file():
            try:
                s = json.loads(status.read_text(encoding="utf-8"))
            except ValueError:
                s = {}
            h = s.get("hierarchy") or {}
            out["final"] = {
                "kept_count": s.get("kept_count"),
                "leaf_total": h.get("leaf_total"),
                "leaf_accepted": h.get("leaf_accepted"),
                "parent_routed": h.get("parent_routed"),
                "top_level_status": s.get("top_level_status"),
            }
        break
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--seeds", type=str, default="")
    ap.add_argument("--corpus", type=int, default=0,
                    help="drive N frozen BENCHMARK_PROMPTS briefs instead of surprise seeds")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--build-timeout-s", type=int, default=2400)
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args(argv)

    corpus_briefs = []
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    elif args.corpus:
        # Surrogate set: the surprise space cannot reach the build (see the
        # routing-handoff plan §8.6), so measure the routing tail on frozen
        # corpus briefs that DO commit their design stages.
        from kicraft.tuning.benchmark import BENCHMARK_PROMPTS

        seeds = []
        corpus_briefs = [e["brief"] for e in BENCHMARK_PROMPTS[: args.corpus]]
    else:
        rng = random.Random()
        seeds = rng.sample(range(BRIEF_SEED_MODULO), args.count)
        corpus_briefs = []

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_root = Path(args.out) if args.out else (
        Path.home() / ".kicraft" / "debug" / f"routing-surprise-{stamp}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"routing surprise cohort: {len(corpus_briefs) if args.corpus and not args.seeds else len(seeds)} briefs -> {out_root}", flush=True)

    from kicraft.server.client import make_client
    from kicraft.server.config import Settings

    settings = Settings.from_env()
    client = make_client(settings)

    results: list[dict] = []
    if corpus_briefs:
        items = [(f"corpus{i:02d}", b) for i, b in enumerate(corpus_briefs, 1)]
    else:
        items = [(str(s), generate_brief(s)) for s in seeds]
    for i, (label, brief) in enumerate(items, 1):
        rundir = out_root / f"{label}-{_slug(brief)}"
        rundir.mkdir(parents=True, exist_ok=True)
        (rundir / "brief.txt").write_text(brief + "\n", encoding="utf-8")
        events = rundir / "events.jsonl"

        def progress(event: dict, _events=events, _label=label) -> None:
            try:
                with open(_events, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, default=str) + "\n")
            except OSError:
                pass
            kind = event.get("kind")
            if kind in {"stage_done", "build_start", "build_done", "top_level_retry"}:
                print(f"  [{_label}] {kind}: "
                      f"{event.get('stage') or event.get('rc') or ''}", flush=True)

        print(f"\n=== {i}/{len(items)} {label} ===\n{brief}", flush=True)
        rec: dict = {"seed": label, "brief": brief, "rundir": str(rundir)}
        t0 = time.time()
        try:
            design = run_design(client, brief, rundir, progress, run_id=str(label))
            rec["design"] = design
        except Exception as exc:  # noqa: BLE001
            rec["design"] = {"status": "exception", "error": repr(exc)}
        if not args.no_build and (rec["design"] or {}).get("status") == "ok":
            rec["build_rc"] = run_build(rundir, progress, timeout_s=args.build_timeout_s)
        rec["routing"] = _routing_summary(rundir)
        rec["duration_s"] = round(time.time() - t0, 1)
        results.append(rec)
        (out_root / "summary.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"  -> design={(rec['design'] or {}).get('status')} "
            f"build_rc={rec.get('build_rc')} "
            f"routing={json.dumps(rec['routing'].get('final') or {})} "
            f"({rec['duration_s']:.0f}s)",
            flush=True,
        )

    print("\n=== cohort summary ===")
    for rec in results:
        fin = (rec.get("routing") or {}).get("final") or {}
        print(
            f"  seed={rec['seed']:<12} design={(rec['design'] or {}).get('status'):<8} "
            f"build_rc={rec.get('build_rc')} "
            f"leaves={fin.get('leaf_accepted')}/{fin.get('leaf_total')} "
            f"parent_routed={fin.get('parent_routed')} "
            f"kept={fin.get('kept_count')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
