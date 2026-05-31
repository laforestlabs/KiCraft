#!/usr/bin/env python3
"""Harvest a finished subject run from its external workspace into a run record.

A CircuitChat eval run happens in a throwaway *workspace* outside the repo
(e.g. ~/kicraft-eval/workspaces/<id>/). When the subject session finishes, this
copies the evidence into a *run record* (~/kicraft-eval/runs/<id>/), locates the
subject's Claude Code transcript, and stamps run.json with provenance so the run
is reproducible. score_run.py then reads the run record.

    .venv/bin/python tests/skill-eval/bin/harvest_run.py \\
        --workspace ~/kicraft-eval/workspaces/S02-20260530-1 \\
        --scenario S02 --target-mode release \\
        [--skill-dir ~/.claude/skills/circuitchat] [--runs-root ~/kicraft-eval/runs]

Copies: .kicraft/, generated/, .claude/settings.local.json. Finds the transcript
under ~/.claude/projects/<mangled-workspace-path>/*.jsonl (newest, or --session).
Nothing is written inside the KiCraft repo.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def mangle_cwd(path: Path) -> str:
    """Claude Code project-dir name: every non-alphanumeric char -> '-'."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(path))


def find_transcript(workspace: Path, session: str | None) -> Path | None:
    proj = Path.home() / ".claude" / "projects" / mangle_cwd(workspace)
    if not proj.is_dir():
        return None
    jsonls = list(proj.glob("*.jsonl"))
    if session:
        for j in jsonls:
            if session in j.stem:
                return j
    return max(jsonls, key=lambda p: p.stat().st_mtime) if jsonls else None


def dir_content_hash(d: Path) -> str | None:
    """Stable sha256 over a directory's file contents (for skill provenance)."""
    if not d or not d.is_dir():
        return None
    h = hashlib.sha256()
    for f in sorted(d.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(d).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def cli_info() -> dict:
    info = {"path": shutil.which("kicraft-circuitchat")}
    try:
        out = subprocess.run(["kicraft-circuitchat", "--help"], capture_output=True,
                             text=True, timeout=30)
        info["help_ok"] = out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        info["help_ok"] = False
    return info


def copy_if(src: Path, dst: Path) -> bool:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return True
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", required=True, help="subject CWD (external)")
    ap.add_argument("--scenario", help="scenario id (e.g. S02)")
    ap.add_argument("--target-mode", choices=["release", "dev"], default="release")
    ap.add_argument("--run-id", help="run record name (default <scenario>-<UTCstamp>)")
    ap.add_argument("--runs-root", default=str(Path.home() / "kicraft-eval" / "runs"))
    ap.add_argument("--skill-dir", help="skill dir under test, for provenance hash")
    ap.add_argument("--session", help="Claude session id substring to disambiguate transcript")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        raise SystemExit(f"workspace not found: {workspace}")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{args.scenario or 'RUN'}-{stamp}"
    run_dir = Path(args.runs_root).expanduser() / run_id
    if run_dir.exists():
        raise SystemExit(f"run record already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    copied = []
    if copy_if(workspace / ".kicraft", run_dir / ".kicraft"):
        copied.append(".kicraft/")
    if copy_if(workspace / "generated", run_dir / "generated"):
        copied.append("generated/")
    if copy_if(workspace / ".claude" / "settings.local.json",
               run_dir / ".claude" / "settings.local.json"):
        copied.append(".claude/settings.local.json")

    transcript = find_transcript(workspace, args.session)
    if transcript:
        shutil.copy2(transcript, run_dir / "transcript.jsonl")
        copied.append(f"transcript.jsonl (from {transcript.name})")

    skill_dir = Path(args.skill_dir).expanduser() if args.skill_dir else None
    kicraft_session = None
    sid = workspace / ".kicraft" / "session_id"
    if sid.is_file():
        kicraft_session = sid.read_text().strip()

    run_meta = {
        "run_id": run_id,
        "scenario": args.scenario,
        "target_mode": args.target_mode,
        "workspace": str(workspace),
        "captured_at": stamp,
        "kicraft_session_id": kicraft_session,
        "claude_transcript": transcript.name if transcript else None,
        "skill_dir": str(skill_dir) if skill_dir else None,
        "skill_sha256": dir_content_hash(skill_dir) if skill_dir else None,
        "cli": cli_info(),
        "copied": copied,
    }
    (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2))

    print(f"harvested -> {run_dir}")
    for c in copied:
        print(f"  + {c}")
    if not transcript:
        print(f"  ! no transcript found under ~/.claude/projects/{mangle_cwd(workspace)}/")
        print("    (latency / #questions / re-commit metrics will be partial)")
    print(f"\nnext: score_run.py score {run_dir}"
          + (f" --scenario {args.scenario}" if args.scenario else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
