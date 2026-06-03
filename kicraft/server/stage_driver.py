"""Generic stage driver: drive KiCraft design stages through the capped client.

This is the agent loop ported out of Claude Code. For each stage it loads the
real stage spec + the slot's Pydantic JSON schema, asks the model (via the
capped gateway) to draft the slot JSON, and commits it with the existing
deterministic CLI. If stage-commit rejects the slot, the validation error is
fed back to the model to self-correct (the same loop the Claude Code skill
runs). Stages chain in one workspace so a project advances
intent -> functional_spec -> architecture -> ... with capped spend throughout.

    python -m kicraft.server.stage_driver \\
        --workspace /tmp/lamp --stages intent,functional_spec,architecture \\
        --brief "a USB-C powered LED night light, no microcontroller"
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from kicraft.design import models

from .client import CappedOpenRouterClient
from .config import Settings

# The repo venv has no `kicraft` console script; cli_app.py has a __main__ guard.
KICRAFT = [sys.executable, "-m", "kicraft.design.cli_app"]

_REPO = Path(__file__).resolve().parents[2]
_SPEC_DIRS = [
    _REPO / "kicraft" / "skill_assets" / "skill" / "stages",   # packaged (preferred)
    _REPO / ".claude" / "skills" / "kicraft" / "stages",       # current dev location
]

# Canonical stage -> slot model, mirroring cli_app._apply_slot's owned-field map.
SLOT_MODEL = {
    "intent": models.IntentSlot,
    "functional_spec": models.FunctionalSpec,
    "architecture": models.Architecture,
    "bom": models.BOM,
}


def _spec_text(stage: str) -> str:
    for d in _SPEC_DIRS:
        p = d / f"{stage}.md"
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return ""  # spec optional; the JSON schema still constrains the output


def build_system(stage: str) -> str:
    spec = _spec_text(stage)
    schema = json.dumps(SLOT_MODEL[stage].model_json_schema())
    extra = ""
    if stage == "intent":
        extra = ('\n- Also include a top-level "project_stem" string (2-3 significant '
                 'words, UPPER_SNAKE_CASE, <=32 chars). It is stripped from the slot and '
                 'passed separately, per the spec.')
    return (
        f"You are the '{stage}' stage of KiCraft, a PCB design assistant running as a "
        f"server (not Claude Code). Draft the '{stage}' slot of the design state.\n\n"
        "Output ONLY a single JSON object: no prose, no markdown fences, no tool calls.\n\n"
        "Follow this stage specification (ignore any references to SKILL.md, sub-agents, "
        "or running CLI tools yourself; you only produce the slot JSON):\n"
        f"=== SPEC ===\n{spec}\n=== END SPEC ===\n\n"
        "The JSON MUST validate against this Pydantic JSON schema (enums, required fields, "
        f"and string patterns are strict):\n{schema}\n\n"
        "Rules:\n"
        "- Output only the slot JSON object.\n"
        "- Use only allowed enum values; honor every naming pattern and uniqueness/"
        "reference constraint.\n"
        '- Every "assumptions" entry must end with "(defaulted)".'
        f"{extra}"
    )


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            text = text[a:b + 1]
    return json.loads(text)


def _fallback_stem(brief: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", brief.upper())[:3]
    return ("_".join(words)[:32]) or "PROJECT"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _commit(stage: str, obj: dict, state_path: Path, brief: str) -> tuple[bool, dict]:
    project_stem = obj.pop("project_stem", None)
    sf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(obj, sf)
    sf.close()
    cmd = KICRAFT + ["stage-commit", stage, "--slot-file", sf.name, str(state_path), "--no-archive"]
    if stage == "intent":
        cmd += ["--project-stem", project_stem or _fallback_stem(brief)]
    proc = _run(cmd)
    Path(sf.name).unlink(missing_ok=True)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        out = {"ok": False, "errors": [proc.stdout.strip() or proc.stderr.strip()]}
    return (proc.returncode == 0 and bool(out.get("ok"))), out


def drive_stage(client, stage, brief, state_path, max_tokens=4096, max_retries=2) -> dict:
    prep = _run(KICRAFT + ["stage-prep", stage, str(state_path)])
    if prep.returncode != 0:
        return {"stage": stage, "commit_ok": False, "cost_usd": 0.0,
                "error": f"stage-prep failed: {prep.stderr.strip()}"}
    prep_json = json.loads(prep.stdout)
    extras = prep_json.get("extras") or {}

    user = (f"PROJECT BRIEF:\n{brief}\n\n"
            f"CURRENT DESIGN STATE (JSON):\n{json.dumps(prep_json['state'])}")
    if extras:
        user += f"\n\nSTAGE EXTRAS (reference data from stage-prep):\n{json.dumps(extras)[:6000]}"
    user += f"\n\nProduce the {stage} slot JSON now."

    messages = [{"role": "system", "content": build_system(stage)},
                {"role": "user", "content": user}]
    total_cost = 0.0
    last: dict = {}
    for attempt in range(max_retries + 1):
        res = client.chat(messages, max_tokens=max_tokens)
        total_cost += res["cost_usd"]
        raw = res["text"] or res.get("reasoning") or ""
        try:
            obj = _extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            last = {"error": "no JSON in reply", "reply_head": raw[:200]}
            messages += [{"role": "assistant", "content": raw[:1000]},
                         {"role": "user", "content": "That was not valid JSON. "
                          "Return ONLY the slot JSON object."}]
            continue

        ok, out = _commit(stage, dict(obj), state_path, brief)
        if ok:
            return {"stage": stage, "commit_ok": True, "cost_usd": total_cost,
                    "attempts": attempt + 1, "commit": out, "slot": obj}
        last = {"commit": out}
        messages += [{"role": "assistant", "content": json.dumps(obj)},
                     {"role": "user", "content":
                      f"stage-commit rejected that with these errors: "
                      f"{json.dumps(out.get('errors'))}. Return a corrected slot JSON "
                      "object only, fixing exactly those."}]
    return {"stage": stage, "commit_ok": False, "cost_usd": total_cost,
            "attempts": max_retries + 1, **last}


def drive_chain(stages, brief, workspace, max_tokens=4096, max_retries=2):
    ws = Path(workspace)
    (ws / ".kicraft").mkdir(parents=True, exist_ok=True)
    state_path = ws / ".kicraft" / "state.json"
    client = CappedOpenRouterClient(Settings.from_env())
    results = []
    for stage in stages:
        r = drive_stage(client, stage, brief, state_path, max_tokens, max_retries)
        results.append(r)
        cost = r.get("cost_usd")
        cstr = f"${cost:.6f}" if isinstance(cost, (int, float)) else "n/a"
        tag = "ok  " if r.get("commit_ok") else "FAIL"
        line = f"  [{tag}] {stage:<16} cost={cstr}  attempts={r.get('attempts', '-')}"
        if not r.get("commit_ok"):
            line += f"\n         -> {r.get('error') or r.get('commit')}"
        print(line)
        if not r.get("commit_ok"):
            break
    return results, client.guard.status(), str(state_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Drive KiCraft stages via the capped gateway.")
    ap.add_argument("--workspace", required=True, help="project dir (holds .kicraft/state.json)")
    ap.add_argument("--brief", required=True, help="the user's project description")
    ap.add_argument("--stages", default="intent",
                    help="comma-separated stages in order (default: intent)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-retries", type=int, default=2,
                    help="self-correction attempts per stage after a rejected commit")
    args = ap.parse_args(argv)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in SLOT_MODEL]
    if bad:
        ap.error(f"unsupported stage(s): {bad}; supported: {sorted(SLOT_MODEL)}")

    print(f"driving {stages} for: {args.brief!r}\n")
    results, guard, state_path = drive_chain(
        stages, args.brief, Path(args.workspace), args.max_tokens, args.max_retries)
    done = [r["stage"] for r in results if r.get("commit_ok")]
    print(f"\ncommitted stages: {done}")
    print(f"total spent: ${guard['spent_total_usd']:.6f}  "
          f"(today remaining ${guard['daily_remaining_usd']:.4f} of ${guard['daily_ceiling_usd']})")
    print(f"state: {state_path}")
    return 0 if results and all(r.get("commit_ok") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
