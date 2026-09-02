"""Stable stage-driving facade and command-line interface."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


from kicraft.fsutil import atomic_write_text
from .stage_pipeline import (
    DESIGN_STAGES,
    SUPPORTED_STAGES,
    drive_chain,
    drive_replay,
    make_budget_client,
    run_pipeline,
)
from .stage_runtime import drive_stage
from .stage_state_io import commit_stage, stamp_stage_status

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

    p_draft = sub.add_parser(
        "debug-draft", help="draft one provider-backed stage without committing it"
    )
    p_draft.add_argument("--workspace", required=True)
    p_draft.add_argument("--stage", required=True, choices=SUPPORTED_STAGES)
    p_draft.add_argument("--brief-file", required=True)
    p_draft.add_argument("--instruction-file")
    p_draft.add_argument("--answers-file")
    p_draft.add_argument("--budget", type=float, default=0.25)
    p_draft.add_argument("--max-tokens", type=int, default=4096)
    p_draft.add_argument("--max-retries", type=int, default=2)
    p_draft.set_defaults(func=_cmd_debug_draft)

    p_commit = sub.add_parser(
        "debug-commit", help="commit an explicitly accepted pending stage candidate"
    )
    p_commit.add_argument("--workspace", required=True)
    p_commit.add_argument("--stage", required=True, choices=SUPPORTED_STAGES)
    p_commit.add_argument("--history-message-file", required=True)
    p_commit.set_defaults(func=_cmd_debug_commit)

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


def _state_sha256(state_path: Path) -> str:
    if not state_path.exists():
        return "absent"
    return hashlib.sha256(state_path.read_bytes()).hexdigest()


def _debug_artifact_path(workspace: Path, stage: str) -> Path:
    return workspace / ".kicraft" / "debug" / f"{stage}.json"


def _write_debug_artifact(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(artifact, indent=2) + "\n")


def _read_text_file(path: str, label: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc


def _read_answers_file(path: str | None) -> list[dict]:
    if path is None:
        return []
    text = _read_text_file(path, "answers-file")
    try:
        answers = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"answers-file must contain valid JSON: {exc}") from exc
    if not isinstance(answers, list) or any(
        not isinstance(answer, dict)
        or set(answer) != {"text", "answer"}
        or not isinstance(answer["text"], str)
        or not isinstance(answer["answer"], str)
        for answer in answers
    ):
        raise ValueError("answers-file must be a JSON list of {text, answer} string objects")
    return answers


def _cmd_debug_draft(args) -> int:
    try:
        if args.budget <= 0:
            raise ValueError("--budget must be greater than zero")
        if args.max_tokens <= 0:
            raise ValueError("--max-tokens must be greater than zero")
        if args.max_retries < 0:
            raise ValueError("--max-retries must be nonnegative")
        workspace = Path(args.workspace)
        state_path = workspace / ".kicraft" / "state.json"
        basis_sha256 = _state_sha256(state_path)
        brief = _read_text_file(args.brief_file, "brief-file")
        instruction = (
            _read_text_file(args.instruction_file, "instruction-file")
            if args.instruction_file
            else None
        )
        answers = _read_answers_file(args.answers_file)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"could not hash state.json: {exc}", file=sys.stderr)
        return 2

    events: list[dict] = []
    try:
        client = make_budget_client(args.budget)
        result = drive_stage(
            client,
            args.stage,
            brief,
            state_path,
            workspace,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            progress=events.append,
            answers=answers or None,
            instruction=instruction,
            review_before_commit=True,
        )
    except Exception as exc:  # provider/config/runtime failures become durable evidence
        result = {
            "stage": args.stage,
            "commit_ok": False,
            "cost_usd": 0.0,
            "attempts": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if result.get("needs_review"):
        status = "needs_review"
    elif result.get("needs_input"):
        status = "needs_input"
    else:
        status = "failed"
    artifact_path = _debug_artifact_path(workspace, args.stage)
    artifact = {
        "version": 1,
        "status": status,
        "stage": args.stage,
        "basis_sha256": basis_sha256,
        "brief": brief,
        "instruction": instruction,
        "answers": answers,
        "result": result,
        "events": events,
    }
    try:
        _write_debug_artifact(artifact_path, artifact)
    except OSError as exc:
        print(f"could not write debug artifact: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "status": status,
                "stage": args.stage,
                "cost_usd": result.get("cost_usd", 0.0),
                "attempts": result.get("attempts", 0),
                "question_count": len(result.get("questions") or []),
                "diagnostic_count": len(result.get("diagnostics") or []),
            },
            separators=(",", ":"),
        )
    )
    return 0 if status in {"needs_review", "needs_input"} else 1


def _cmd_debug_commit(args) -> int:
    workspace = Path(args.workspace)
    state_path = workspace / ".kicraft" / "state.json"
    artifact_path = _debug_artifact_path(workspace, args.stage)
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        history_message = _read_text_file(
            args.history_message_file, "history-message-file"
        ).strip()
        if not history_message:
            raise ValueError("history-message-file must not be empty")
        if artifact.get("version") != 1:
            raise ValueError("debug artifact version must be 1")
        if artifact.get("stage") != args.stage:
            raise ValueError("debug artifact stage does not match --stage")
        if artifact.get("status") != "needs_review":
            raise ValueError("debug artifact status must be needs_review")
        result = artifact.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("slot"), dict):
            raise ValueError("debug artifact has no review candidate slot")
        current_sha256 = _state_sha256(state_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if current_sha256 != artifact.get("basis_sha256"):
        print("state changed since draft; re-run debug-draft", file=sys.stderr)
        return 2

    slot = dict(result["slot"])
    project_stem = slot.pop("project_stem", None) if args.stage == "intent" else None
    ok, commit_result = commit_stage(
        args.stage,
        slot,
        state_path,
        artifact.get("brief") or "",
        project_stem,
        workspace,
        invalidate_downstream=True,
        history_message=history_message,
    )
    if not ok:
        print(
            json.dumps(
                {
                    "ok": False,
                    "stage": args.stage,
                    "errors": commit_result.get("errors") or [],
                    "offenders": commit_result.get("offenders") or [],
                },
                separators=(",", ":"),
            )
        )
        return 1

    stamp_stage_status(
        state_path,
        args.stage,
        True,
        cost_usd=result.get("cost_usd"),
        attempts=result.get("attempts"),
        rounds=result.get("rounds"),
        tool_calls=result.get("tool_calls"),
        wall_s=result.get("wall_s"),
        cpu_s=result.get("cpu_s"),
        provider_ok=result.get("provider_ok"),
        schema_ok=result.get("schema_ok"),
        semantic_clean=result.get("semantic_clean"),
        repair_required=result.get("repair_required", False),
        fab_safe=result.get("fab_safe"),
        diagnostics=result.get("diagnostics") or [],
    )
    accepted_sha256 = _state_sha256(state_path)
    artifact["status"] = "accepted"
    artifact["accepted_state_sha256"] = accepted_sha256
    artifact["commit"] = commit_result
    try:
        _write_debug_artifact(artifact_path, artifact)
    except OSError as exc:
        print(f"candidate committed but debug artifact update failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "status": "accepted",
                "stage": args.stage,
                "accepted_state_sha256": accepted_sha256,
                "invalidated_stages": commit_result.get("invalidated_stages") or [],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
