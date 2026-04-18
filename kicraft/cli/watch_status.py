#!/usr/bin/env python3
"""Live viewer for autoexperiment run_status.json.

Usage:
    python3 watch_status.py
    python3 watch_status.py --file ../../../../.experiments/run_status.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def _format_mmss(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    try:
        s = max(0, int(float(seconds)))
    except (ValueError, TypeError):
        return "n/a"
    return f"{s // 60}m{s % 60:02d}s"


def _num(v, precision: int = 2) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{precision}f}"
    except (ValueError, TypeError):
        return "n/a"


def _clear() -> None:
    # ANSI clear screen + home cursor; works on most terminals.
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def _read_status(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _render(status: dict, path: Path, age_s: float) -> str:
    workers = status.get("workers", {}) if isinstance(status.get("workers"), dict) else {}
    phase = status.get("phase", "unknown")
    health = "MAYBE STUCK" if status.get("maybe_stuck") else "active"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=== Autoexperiment Live Monitor ===",
        f"time:      {now}",
        f"source:    {path}",
        f"snapshot:  {_format_mmss(age_s)} old",
        "",
        f"phase:     {phase}",
        f"progress:  {status.get('round', 'n/a')}/{status.get('total_rounds', 'n/a')} ({_num(status.get('progress_percent'), 1)}%)",
        f"workers:   total={workers.get('total', 'n/a')} in_flight={workers.get('in_flight', 'n/a')} idle={workers.get('idle', 'n/a')}",
        "",
        f"best:      {_num(status.get('best_score'))}",
        f"latest:    {_num(status.get('latest_score'))}  [{status.get('latest_marker', 'n/a')}]",
        f"kept:      {status.get('kept_count', 'n/a')}",
        f"stagnant:  {status.get('minor_stagnant', 'n/a')}",
        "",
        f"elapsed:   {_format_mmss(status.get('elapsed_s'))}",
        f"eta:       {_format_mmss(status.get('eta_s'))}",
        f"avg round: {_num(status.get('avg_round_s'), 1)}s",
        f"speed:     {_num(status.get('throughput_rounds_per_min'))} rounds/min",
        "",
        f"since last completion: {_format_mmss(status.get('time_since_last_completion_s'))}",
        f"idle threshold:        {_format_mmss(status.get('idle_threshold_s'))}",
        f"health:                {health}",
        "",
        "Press Ctrl+C to exit.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch autoexperiment live status file.")
    parser.add_argument(
        "--file",
        "-f",
        default="../../../../.experiments/run_status.json",
        help="Path to run_status.json",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=1.0,
        help="Refresh interval in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    status_path = Path(args.file).expanduser().resolve()
    interval = max(0.2, args.interval)

    try:
        while True:
            status = _read_status(status_path)
            _clear()
            if status is None:
                print("=== Autoexperiment Live Monitor ===")
                print(f"source: {status_path}")
                print("")
                print("No readable status yet.")
                print("Start autoexperiment first, or check --file path.")
                print("")
                print("Press Ctrl+C to exit.")
            else:
                try:
                    mtime = status_path.stat().st_mtime
                    age_s = max(0.0, time.time() - mtime)
                except OSError:
                    age_s = 0.0
                print(_render(status, status_path, age_s))
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
