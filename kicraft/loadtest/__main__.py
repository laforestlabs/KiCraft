"""CLI for the load-test harness.

    python -m kicraft.loadtest build-storm --n 12 --slots 2 [--route]
    python -m kicraft.loadtest pipeline    --n 8 --parallel 3 --build-slots 2 [--no-build]
    python -m kicraft.loadtest web                # prints the external-driver recipe

Build-storm and pipeline are $0 (replay / mock LLM); only a separately-run live
smoke spends. Results land in the LoadResultStore and surface on /admin/loadtest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import harness, scenarios
from .mockllm import load_transcript


def _cmd_build_storm(args) -> int:
    source = Path(args.source) if args.source else scenarios.find_synth_workspace()
    if source is None:
        print("no synthesized workspace found; pass --source DIR (a finished run dir "
              "with .kicraft/state.json + generated/<stem>)", file=sys.stderr)
        return 2
    print(f"[build-storm] source={source} n={args.n} slots={args.slots} "
          f"route={args.route}", flush=True)
    out = harness.run_build_storm(
        source=source, n=args.n, slots=args.slots, route=args.route,
        store_path=Path(args.store) if args.store else None,
        interval_s=args.interval, timeout_s=args.build_timeout,
        abort_file=args.abort_file)
    _print_summary(out)
    return 0


def _cmd_pipeline(args) -> int:
    transcript = load_transcript(args.transcript or scenarios.DEFAULT_TRANSCRIPT)
    briefs = [scenarios.DEFAULT_BRIEF] * args.n
    print(f"[pipeline] n={args.n} parallel={args.parallel} build_slots={args.build_slots} "
          f"build={not args.no_build}", flush=True)
    out = harness.run_pipeline(
        briefs=briefs, parallel=args.parallel, build_slots=args.build_slots,
        transcript=transcript, do_build=not args.no_build,
        store_path=Path(args.store) if args.store else None,
        interval_s=args.interval, build_timeout_s=args.build_timeout)
    _print_summary(out)
    return 0


def _cmd_web(args) -> int:
    print(
        "Web-tier load is driven by external tools (NiceGUI is websocket-heavy):\n"
        "  HTTP routes:   k6 run scripts/loadtest_web.js  -e BASE=http://127.0.0.1:8080\n"
        "  authed session: locust -f scripts/loadtest_web_ws.py --host http://127.0.0.1:8080\n"
        "  signed file token (no prod route added):\n"
        "      .venv/bin/python scripts/mint_loadtest_token.py <project_dir>\n"
        "Run the web app with KICRAFT_LLM_MODE=mock + KICRAFT_MOCK_TRANSCRIPT so design\n"
        "submits exercise the pipeline at $0.")
    return 0


def _print_summary(out: dict) -> None:
    s = out["summary"]
    print(f"\nrun_id={out['run_id']}  store={out['store']}")
    print(json.dumps({k: v for k, v in s.items() if k not in ("jobs", "records")}, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kicraft.loadtest",
                                 description="KiCraft load / stress harness")
    ap.add_argument("--store", help="LoadResultStore path (default ~/.kicraft/loadtest/loadtest.db)")
    ap.add_argument("--interval", type=float, default=1.0, help="metrics sample cadence (s)")
    ap.add_argument("--build-timeout", type=float, default=1800.0)
    sub = ap.add_subparsers(dest="cmd", required=True)

    bs = sub.add_parser("build-storm", help="enqueue N replay builds, measure the queue")
    bs.add_argument("--source", help="synthesized workspace to clone (auto-discovered if omitted)")
    bs.add_argument("--n", type=int, default=12)
    bs.add_argument("--slots", type=int, default=2)
    bs.add_argument("--route", action="store_true", help="route (heavy); default placement-only")
    bs.add_argument("--abort-file", help="abort the storm if this file appears")
    bs.set_defaults(func=_cmd_build_storm)

    pl = sub.add_parser("pipeline", help="full LLM pipeline at $0 via the mock client")
    pl.add_argument("--n", type=int, default=8)
    pl.add_argument("--parallel", type=int, default=3)
    pl.add_argument("--build-slots", type=int, default=2)
    pl.add_argument("--no-build", action="store_true", help="design stages only, skip the build")
    pl.add_argument("--transcript", help="transcript JSON (default: committed fixture)")
    pl.set_defaults(func=_cmd_pipeline)

    web = sub.add_parser("web", help="print the external web-load driver recipe")
    web.set_defaults(func=_cmd_web)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
