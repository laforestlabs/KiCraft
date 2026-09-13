"""Run one arm of the architecture correction-ladder experiment.

Operator harness for `docs/plans/architecture-contract-correction-ladder.md`
§3: replay the frozen pre-architecture state of a board through the REAL
provider and stage pipeline once per run, under exactly one
`KICRAFT_CONTRACT_LADDER` arm, and write every artifact the plan's metrics need.

Per run it records, into `<out>/runs.jsonl`:

  commit / failure_kind / attempts / rungs (the call-mode ladder)
  defects per rung (the blocking diagnostic codes the drive was shown)
  dropped declared content (nets/ports present in rung i, absent in rung i+1
  with no diagnostic naming them)
  cost, wall seconds, and the raw event + attempt traces (separate files)

Usage (one arm at a time; arms share the process-wide spend guard):

    .venv/bin/python tools/ladder_experiment.py \
        --arm preserving --state ~/.kicraft/projects/1/825/.kicraft/state.json \
        --runs 10 --out /tmp/ladder-exp

    .venv/bin/python tools/ladder_experiment.py --summary /tmp/ladder-exp
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kicraft.server.config import CONTRACT_LADDER_MODES, Settings  # noqa: E402
from kicraft.server.stage_pipeline import drive_replay  # noqa: E402


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _diagnostic_codes(event: dict) -> list[str]:
    row = event.get("diagnostic") or {}
    codes = [str(row["code"])] if row.get("code") else []
    codes.extend(
        str(item["code"])
        for item in row.get("evidence") or []
        if isinstance(item, dict) and item.get("code")
    )
    return codes


def _rejection_text(event: dict) -> str:
    return " ".join(
        str(part)
        for part in (event.get("schema_error"), json.dumps(event.get("diagnostic") or {}))
        if part
    )


def _identity_name(identity: str) -> str:
    return identity.split(":", 1)[1]


def _run_metrics(stage_result: dict, events: list[dict], trace: list[dict]) -> dict:
    retries = [event for event in events if event.get("kind") == "retry"]
    identities = [
        event.get("declared_identities") or []
        for event in retries
        if event.get("declared_identities") is not None
    ]
    dropped: list[dict] = []
    for index in range(1, len(identities)):
        previous, current = set(identities[index - 1]), set(identities[index])
        text = _rejection_text(retries[index])
        lost = sorted(
            _identity_name(item)
            for item in previous - current
            if _identity_name(item) not in text
        )
        if lost:
            dropped.append({"rung": index + 1, "items": lost})
    return {
        "commit": bool(stage_result and stage_result.get("commit_ok")),
        "failure_kind": (stage_result or {}).get("failure_kind"),
        "attempts": (stage_result or {}).get("attempts"),
        "rungs": [
            {"attempt": row.get("provider_attempt"), "mode": row.get("call_mode"),
             "outcome": row.get("outcome")}
            for row in trace
        ],
        "defects": [
            {
                "attempt": row.get("provider_attempt"),
                "call_mode": row.get("call_mode"),
                "failure_kind": row.get("failure_kind"),
                "codes": _diagnostic_codes(row),
            }
            for row in retries
        ],
        "dropped_content": dropped,
        "cost_usd": (stage_result or {}).get("cost_usd"),
        "wall_s": (stage_result or {}).get("wall_s"),
    }


def _spent_total(owner) -> float:
    guard = getattr(owner, "guard", None)
    if guard is None:
        return 0.0
    status = guard.status() if callable(getattr(guard, "status", None)) else {}
    return float((status or {}).get("spent_total_usd") or 0.0)


def run_arm(args) -> int:
    if args.arm != "stock" and args.arm not in CONTRACT_LADDER_MODES:
        print(f"unknown arm {args.arm!r}; one of {sorted(CONTRACT_LADDER_MODES)}", file=sys.stderr)
        return 2
    os.environ["KICRAFT_CONTRACT_LADDER"] = args.arm
    label = args.label or args.arm
    state = Path(args.state).expanduser()
    board = args.board or state.parent.parent.name
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    manifest = {
        "arm": args.arm,
        "label": label,
        "board": board,
        "state": str(state),
        "runs_requested": args.runs,
        "budget_per_run_usd": args.budget,
        "max_retries": args.max_retries,
        "ceiling_usd": args.ceiling,
        "repo_head": _git("rev-parse", "HEAD"),
        "repo_dirty": bool(_git("status", "--porcelain")),
        "contract_ladder": os.environ["KICRAFT_CONTRACT_LADDER"],
        "settings": Settings.from_env().redacted(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / f"manifest-{label}-{board}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"arm={args.arm} board={board} runs={args.runs} budget=${args.budget} "
        f"ceiling=${args.ceiling} head={manifest['repo_head'][:12]}"
    )

    spent_start = None
    committed = 0
    completed = 0
    with runs_path.open("a", encoding="utf-8") as sink:
        for index in range(1, args.runs + 1):
            events: list[dict] = []
            trace: list[dict] = []
            from kicraft.server.stage_pipeline import make_budget_client

            client = make_budget_client(args.budget)
            if spent_start is None:
                spent_start = _spent_total(client)
            started = time.monotonic()
            # One unique artifact pair per run: an interleaved driver invokes this
            # script once per run with `--runs 1`, so a label+index name would
            # overwrite the previous run's events and trace.
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            events_path = out / f"{label}-{board}-{stamp}.events.jsonl"
            trace_path = out / f"{label}-{board}-{stamp}.trace.jsonl"
            record: dict = {
                "arm": args.arm,
                "label": label,
                "board": board,
                "run": index,
                "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "artifacts": {"events": str(events_path), "trace": str(trace_path)},
            }
            try:
                result = drive_replay(
                    state,
                    "architecture",
                    budget_usd=args.budget,
                    max_retries=args.max_retries,
                    progress=events.append,
                    attempt_observer=trace.append,
                    client=client,
                )
                events_path.write_text(
                    "".join(
                        json.dumps(event, separators=(",", ":")) + "\n" for event in events
                    ),
                    encoding="utf-8",
                )
                trace_path.write_text(
                    "".join(
                        json.dumps(row, separators=(",", ":")) + "\n" for row in trace
                    ),
                    encoding="utf-8",
                )
                record.update(_run_metrics(result.get("stage"), events, trace))
                record["error"] = result.get("error")
            except Exception as exc:  # noqa: BLE001 - every aborted run is reported
                record.update(
                    {
                        "commit": False,
                        "aborted": True,
                        "error": f"{type(exc).__name__}: {exc}",
                        "rungs": [
                            {"attempt": row.get("provider_attempt"), "mode": row.get("call_mode"),
                             "outcome": row.get("outcome")}
                            for row in trace
                        ],
                    }
                )
            record["wall_s_total"] = round(time.monotonic() - started, 3)
            record["spent_total_usd"] = _spent_total(client)
            record["spent_arm_usd"] = round(record["spent_total_usd"] - spent_start, 6)
            completed += 1
            committed += 1 if record.get("commit") else 0
            sink.write(json.dumps(record, separators=(",", ":")) + "\n")
            sink.flush()
            print(
                f"  r{index:<3} {'COMMIT' if record.get('commit') else 'fail  '} "
                f"{record.get('failure_kind') or '-'} attempts={record.get('attempts')} "
                f"cost=${record.get('cost_usd') or 0:.4f} arm=${record['spent_arm_usd']:.4f}",
                flush=True,
            )
            if record.get("spent_arm_usd", 0.0) >= args.ceiling:
                print(f"  stop: arm reached the ${args.ceiling} ceiling")
                break
    print(f"arm {args.arm}: {committed}/{completed} committed")
    return 0


def summarise(out: Path) -> int:
    runs_path = out / "runs.jsonl"
    if not runs_path.is_file():
        print(f"no {runs_path}", file=sys.stderr)
        return 2
    rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()]
    arms: dict[str, dict] = {}
    for row in rows:
        key = str(row.get("label") or row["arm"])
        bucket = arms.setdefault(key, {"runs": 0, "commits": 0, "aborted": 0, "cost": 0.0, "boards": {}})
        bucket["runs"] += 1
        bucket["commits"] += 1 if row.get("commit") else 0
        bucket["aborted"] += 1 if row.get("aborted") else 0
        bucket["cost"] += float(row.get("cost_usd") or 0.0)
        board = bucket["boards"].setdefault(row["board"], {"runs": 0, "commits": 0})
        board["runs"] += 1
        board["commits"] += 1 if row.get("commit") else 0
    print(f"{'arm':<18} {'commits/runs':>12} {'pooled':>8} {'cost':>9}  per board")
    for arm, bucket in sorted(arms.items()):
        pooled = bucket["commits"] / bucket["runs"] if bucket["runs"] else 0.0
        per_board = "  ".join(
            f"{board}:{value['commits']}/{value['runs']}"
            for board, value in sorted(bucket["boards"].items())
        )
        print(
            f"{arm:<18} {bucket['commits']:>5}/{bucket['runs']:<6} {pooled:>8.2f} "
            f"${bucket['cost']:>8.4f}  {per_board}"
        )
    return 0


def scan_corpus(roots: list[Path], since: float | None = None) -> int:
    """Count terminal-by-policy stage deaths across local event streams.

    Uses the production reader (`triage.collect_stages`) rather than a second
    detector, so the number matches what `triage stages` reports for one board.
    """
    from kicraft.cli.triage import collect_stages, resolve_projects_dir

    projects = roots or [resolve_projects_dir()]
    streams = sorted(
        path
        for root in projects
        for pattern in ("*/*/events.jsonl", "*/events.jsonl")
        for path in root.glob(pattern)
    )
    counts: dict[str, dict[str, int]] = {}
    scanned = 0
    for stream in streams:
        if since is not None and stream.stat().st_mtime < since:
            continue
        run = stream.parent
        try:
            stages = collect_stages(run)
        except Exception as exc:  # noqa: BLE001 - one unreadable run must not stop the scan
            print(f"  skipped {run}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        scanned += 1
        for row in stages.get("stages") or []:
            stage = str(row.get("stage"))
            bucket = counts.setdefault(stage, {"runs": 0, "terminal_by_policy": 0})
            bucket["runs"] += 1
            if not row.get("ok") and row.get("terminal_by_policy"):
                bucket["terminal_by_policy"] += 1
    print(f"scanned {scanned} event streams under {[str(root) for root in projects]}")
    print(f"{'stage':<18} {'terminal_by_policy':>19} {'streams':>8}")
    for stage, bucket in sorted(counts.items()):
        print(f"{stage:<18} {bucket['terminal_by_policy']:>19} {bucket['runs']:>8}")
    return 0


def run_full(args) -> int:
    """Drive all five stages + the deterministic build for a fresh brief."""
    from kicraft.server.stage_pipeline import run_pipeline
    from kicraft.server.stage_state import DESIGN_STAGES

    if args.arm != "stock" and args.arm not in CONTRACT_LADDER_MODES:
        print(f"unknown arm {args.arm!r}", file=sys.stderr)
        return 2
    os.environ["KICRAFT_CONTRACT_LADDER"] = args.arm
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    brief = args.brief or (Path(args.state).expanduser().parent.parent / "brief.txt").read_text(
        encoding="utf-8"
    )
    for index in range(1, args.runs + 1):
        workspace = out / f"full-{args.arm}-r{index}"
        result = run_pipeline(
            brief.strip(),
            workspace,
            stages=DESIGN_STAGES,
            budget_usd=args.budget,
            max_retries=args.max_retries,
            build=True,
        )
        record = {
            "arm": args.arm,
            "run": index,
            "workspace": str(workspace),
            "all_committed": result["all_committed"],
            "build_rc": result["build_rc"],
            "cost_usd": sum(
                float(stage.get("cost_usd") or 0.0) for stage in result["stages"]
            ),
            "stages": [
                {
                    "stage": stage.get("stage"),
                    "commit_ok": stage.get("commit_ok"),
                    "attempts": stage.get("attempts"),
                    "failure_kind": stage.get("failure_kind"),
                }
                for stage in result["stages"]
            ],
        }
        (out / "full-runs.jsonl").open("a", encoding="utf-8").write(
            json.dumps(record, separators=(",", ":")) + "\n"
        )
        print(
            f"  full r{index}: all_committed={record['all_committed']} "
            f"build_rc={record['build_rc']} cost=${record['cost_usd']:.4f}"
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", help="the KICRAFT_CONTRACT_LADDER arm to run")
    parser.add_argument("--state", help="frozen state.json (intent+functional_spec committed)")
    parser.add_argument("--board", help="board label for the artifact names (default: project id)")
    parser.add_argument(
        "--label",
        help="scoreboard label for these runs (default: the arm name); use it to keep a "
        "re-measurement or a knob variant separate from the arm's round-1 rows",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--budget", type=float, default=0.25, help="per-run USD cap")
    parser.add_argument("--max-retries", type=int, default=3, help="the production caller value")
    parser.add_argument(
        "--ceiling", type=float, default=10.0, help="hard stop on the arm's cumulative spend"
    )
    parser.add_argument("--out", default="/tmp/ladder-exp", help="artifact directory")
    parser.add_argument("--summary", action="store_true", help="print the arm scoreboard and exit")
    parser.add_argument(
        "--full",
        action="store_true",
        help="drive all five stages + build (L2) instead of the architecture replay",
    )
    parser.add_argument("--brief", help="brief text for --full (default: the board's brief.txt)")
    parser.add_argument(
        "--since",
        help="--scan only: ignore event streams older than this ISO-8601 timestamp",
    )
    parser.add_argument(
        "--scan",
        nargs="*",
        metavar="PROJECTS_DIR",
        help="count terminal-by-policy stage deaths across local event streams and exit",
    )
    args = parser.parse_args(argv)
    if args.scan is not None:
        since = (
            datetime.fromisoformat(args.since).timestamp() if args.since else None
        )
        return scan_corpus([Path(item).expanduser() for item in args.scan], since)
    if args.summary:
        return summarise(Path(args.out).expanduser())
    if not args.arm:
        parser.error("--arm is required unless --summary/--scan is given")
    if args.full:
        return run_full(args)
    if not args.state:
        parser.error("--state is required for an architecture replay")
    return run_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())
