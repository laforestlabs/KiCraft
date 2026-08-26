"""Stable stage-driving facade and command-line interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .stage_pipeline import (
    DESIGN_STAGES,
    SUPPORTED_STAGES,
    drive_chain,
    drive_replay,
    make_budget_client,
    run_pipeline,
)
from .stage_runtime import drive_stage

__all__ = [
    "DESIGN_STAGES",
    "drive_stage",
    "drive_chain",
    "run_pipeline",
    "drive_replay",
    "make_budget_client",
]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="kicraft.stage_driver",
        description="Drive KiCraft design stages through the capped gateway.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run", help="drive the LLM design stages (optionally + build) from a brief"
    )
    p_run.add_argument("--brief", required=True, help="the user's project description")
    p_run.add_argument("--workspace", required=True, help="project dir (holds .kicraft/state.json)")
    p_run.add_argument(
        "--stages", default=",".join(DESIGN_STAGES), help="comma-separated stages in order"
    )
    p_run.add_argument("--max-tokens", type=int, default=4096)
    p_run.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="self-correction attempts per stage after a rejected commit",
    )
    p_run.add_argument(
        "--budget", type=float, default=0.25, help="per-run USD cap on LLM spend (default $0.25)"
    )
    p_run.add_argument(
        "--no-build",
        action="store_true",
        help="stop after the LLM stages (skip the deterministic build)",
    )
    p_run.add_argument("--quality", choices=["fast", "draft", "good", "best"], default="good")
    p_run.set_defaults(func=_cmd_run)

    p_replay = sub.add_parser(
        "replay", help="re-run ONE LLM stage from a frozen, committed state.json"
    )
    p_replay.add_argument("--state", required=True, help="path to a committed state.json")
    p_replay.add_argument(
        "--stage", required=True, help=f"stage to re-drive; one of {list(SUPPORTED_STAGES)}"
    )
    p_replay.add_argument("--max-retries", type=int, default=2)
    p_replay.add_argument("--budget", type=float, default=0.25)
    p_replay.set_defaults(func=_cmd_replay)

    args = ap.parse_args(argv)
    return args.func(args)


def _cmd_run(args) -> int:
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in SUPPORTED_STAGES]
    if bad:
        print(f"unsupported stage(s): {bad}; supported: {list(SUPPORTED_STAGES)}", file=sys.stderr)
        return 2
    print(f"driving {stages} (LLM budget ${args.budget:.2f}) for: {args.brief!r}\n")
    out = run_pipeline(
        args.brief,
        Path(args.workspace),
        stages=stages,
        budget_usd=args.budget,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        build=not args.no_build,
        quality=args.quality,
    )
    guard = out["guard"]
    print(f"\ncommitted stages: {'all' if out['all_committed'] else 'partial/failed'}")
    print(f"build rc: {out['build_rc'] if out['build_rc'] is not None else 'skipped'}")
    print(
        f"total spent: ${guard['spent_total_usd']:.6f}  "
        f"(today remaining ${guard['daily_remaining_usd']:.4f})"
    )
    print(f"state: {out['state_path']}")
    return 0 if out["all_committed"] else 1


def _cmd_replay(args) -> int:
    print(f"replaying stage {args.stage!r} from {args.state!r} (LLM budget ${args.budget:.2f})\n")
    out = drive_replay(args.state, args.stage, budget_usd=args.budget, max_retries=args.max_retries)
    if "error" in out:
        print(f"replay failed: {out['error']}", file=sys.stderr)
        return 2
    # drive_chain already printed the per-stage [ok/FAIL] line; only add the
    # replay-specific footer here.
    print(f"\nworkspace: {out['workspace']}  (source state untouched)")
    print(f"state: {out['state_path']}")
    return 0 if out["all_committed"] else 1


if __name__ == "__main__":
    sys.exit(main())
