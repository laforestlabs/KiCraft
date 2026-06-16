#!/usr/bin/env python3
"""Per-board summary of a tuning run's eval store.

Reads a tuning ``tuning.db`` (the sqlite results cache the auto-tuner writes)
and prints one row per board across ALL evaluated configs, worst-first by how
often the board produced an EMPTY board (0 routed traces). Use it to spot the
boards that drag the objective: a board whose ``drcMin`` is still the
missing-board sentinel (999) never routed under any config tried, so it is
unroutable as-frozen and just skews the reward.

    python scripts/tboard.py [path/to/tuning.db]

Defaults to the homelab i8 run location so it can be run inside the tuning
container with no extra argument (keeps the docker-exec command line short).
"""
from __future__ import annotations

import sqlite3
import statistics
import sys
from collections import defaultdict

DEFAULT_DB = "/data/runs/i8/tuning.db"


def main(argv: list[str]) -> int:
    db_path = argv[1] if len(argv) > 1 else DEFAULT_DB
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    by_board: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in conn.execute(
        "SELECT board, fab_ready, drc_total, traces, wall_s FROM evals"
    ):
        by_board[r["board"]].append(r)
    if not by_board:
        print(f"no evals in {db_path}")
        return 1

    def empty_frac(rows: list[sqlite3.Row]) -> float:
        return sum((r["traces"] or 0) == 0 for r in rows) / len(rows)

    print(f"{'board':26s} {'n':>3} {'empty':>7} {'fab%':>5} "
          f"{'drcMed':>7} {'drcMin':>7} {'wallMed':>7}")
    n_total = empty_total = 0
    for b in sorted(by_board, key=lambda b: -empty_frac(by_board[b])):
        rows = by_board[b]
        n = len(rows)
        empty = sum((r["traces"] or 0) == 0 for r in rows)
        fab = 100.0 * sum(bool(r["fab_ready"]) for r in rows) / n
        drcs = [r["drc_total"] for r in rows]
        walls = [r["wall_s"] for r in rows]
        n_total += n
        empty_total += empty
        print(f"{b:26s} {n:>3} {empty:>3}/{n:<3} {fab:5.0f} "
              f"{statistics.median(drcs):7.0f} {min(drcs):7.0f} "
              f"{statistics.median(walls):7.0f}")
    print(f"\n{len(by_board)} boards, {n_total} evals, "
          f"{empty_total} empty ({100.0 * empty_total / n_total:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
