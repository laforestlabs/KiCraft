"""Generic stage driver: drive KiCraft design stages through the capped client.

This is the agent loop ported out of Claude Code. For each stage it loads the
real stage spec + the slot's Pydantic JSON schema, asks the model (via the
capped gateway) to draft the slot JSON, and commits it with the existing
deterministic CLI. If stage-commit rejects the slot, the validation error is
fed back to the model to self-correct (the same loop the Claude Code skill
runs). The BOM stage additionally gets tools (list_parts / lookup_symbol /
lookup_lcsc_id / add_part_from_lcsc) so the model resolves real symbols and
footprints (or fetches them from LCSC) instead of guessing.

    python -m kicraft.server.stage_driver \\
        --workspace /tmp/lamp --stages intent,functional_spec,architecture,bom \\
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

# Tools exposed to the model during the BOM stage (OpenAI tool-spec form).
BOM_TOOLS = [
    {"type": "function", "function": {
        "name": "list_parts",
        "description": "List curated parts-library bundles available to this project "
                       "(vendored + any fetched). Returns a table with the exact symbol and "
                       "footprint strings to use verbatim in the BOM.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "lookup_symbol",
        "description": "Verify a KiCad symbol in 'Library:Name' form exists and list its pins.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "e.g. 'Device:R' or "
                       "'usb-c-16p:TYPE-C-31-M-12'"}}, "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "lookup_lcsc_id",
        "description": "Resolve a manufacturer part number or search keyword to an LCSC part "
                       "number (C#####). Returns {ok, lcsc} or a candidates list.",
        "parameters": {"type": "object", "properties": {
            "mpn": {"type": "string", "description": "MPN or keyword, e.g. 'SK-12D07VG4' or "
                    "'SPDT slide switch SMD'"}}, "required": ["mpn"]}}},
    {"type": "function", "function": {
        "name": "add_part_from_lcsc",
        "description": "Fetch a real symbol+footprint bundle from LCSC into the project parts "
                       "library. Afterwards call list_parts to get the exact '<name>:<symbol>' "
                       "and '<name>:<footprint>' strings.",
        "parameters": {"type": "object", "properties": {
            "lcsc_id": {"type": "string", "description": "LCSC part number like C2837270"},
            "name": {"type": "string", "description": "optional library slug"}},
            "required": ["lcsc_id"]}}},
]


def _spec_text(stage: str) -> str:
    for d in _SPEC_DIRS:
        p = d / f"{stage}.md"
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return ""


def _stage_extra(stage: str) -> str:
    if stage == "intent":
        return ('\n- Also include a top-level "project_stem" string (2-3 significant words, '
                'UPPER_SNAKE_CASE, <=32 chars). It is stripped from the slot and passed '
                'separately, per the spec.')
    if stage == "bom":
        return (
            "\n- TOOLS available this stage: list_parts (curated bundles + exact symbol/"
            "footprint strings); lookup_symbol (verify a 'Library:Name' symbol exists + pins); "
            "lookup_lcsc_id (MPN/keyword -> LCSC C-number); add_part_from_lcsc (fetch a real "
            "symbol+footprint bundle into the project).\n"
            "- Use a library bundle VERBATIM when one matches (e.g. usb-c-16p for a USB-C "
            "receptacle): symbol '<name>:<sym>', footprint '<name>:<fp>'.\n"
            "- For any connector/switch/IC NOT in the library and NOT a trivial passive, DO NOT "
            "guess a footprint name. Resolve it: lookup_lcsc_id then add_part_from_lcsc, then "
            "list_parts to read the exact strings.\n"
            "- For trivial passives (R, C, L, LED, diode) use stock KiCad: Device:R / Device:LED "
            "/ Device:C with Resistor_SMD / LED_SMD / Capacitor_SMD footprints (e.g. "
            "'Resistor_SMD:R_0603_1608Metric', 'LED_SMD:LED_0603_1608Metric').\n"
            "- Every symbol AND footprint MUST resolve to a real file. When finished, output "
            "ONLY the BOM slot JSON.")
    return ""


def build_system(stage: str) -> str:
    spec = _spec_text(stage)
    schema = json.dumps(SLOT_MODEL[stage].model_json_schema())
    return (
        f"You are the '{stage}' stage of KiCraft, a PCB design assistant running as a server "
        f"(not Claude Code). Draft the '{stage}' slot of the design state.\n\n"
        "Output ONLY a single JSON object: no prose, no markdown fences.\n\n"
        "Follow this stage specification (ignore references to SKILL.md or sub-agents; you "
        "produce the slot JSON and may use the listed tools):\n"
        f"=== SPEC ===\n{spec}\n=== END SPEC ===\n\n"
        "The JSON MUST validate against this Pydantic JSON schema (enums, required fields, and "
        f"string patterns are strict):\n{schema}\n\n"
        "Rules:\n"
        "- Output only the slot JSON object.\n"
        "- Use only allowed enum values; honor every naming pattern and uniqueness/reference "
        "constraint.\n"
        '- Every "assumptions" entry must end with "(defaulted)".'
        f"{_stage_extra(stage)}"
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


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=(str(cwd) if cwd else None))


def _bom_executor(workspace: Path):
    """Return an executor(name, args) -> str backed by the kicraft CLI (cwd=workspace)."""
    def execute(name: str, args: dict) -> str:
        if name == "list_parts":
            r = _run(KICRAFT + ["list-parts"], workspace)
            return (r.stdout or r.stderr)[:8000]
        if name == "lookup_symbol":
            r = _run(KICRAFT + ["lookup-symbol", str(args.get("symbol", ""))], workspace)
            return (r.stdout or r.stderr)[:3000]
        if name == "lookup_lcsc_id":
            r = _run(KICRAFT + ["lookup-lcsc-id", str(args.get("mpn", ""))], workspace)
            return (r.stdout or r.stderr)[:3000]
        if name == "add_part_from_lcsc":
            cmd = ["add-part", "--from-lcsc", str(args.get("lcsc_id", "")), "--into", "project"]
            if args.get("name"):
                cmd += ["--name", str(args["name"])]
            r = _run(KICRAFT + cmd, workspace)
            lp = _run(KICRAFT + ["list-parts"], workspace)
            return (f"add-part exit={r.returncode}\n{(r.stdout + chr(10) + r.stderr).strip()[:1500]}"
                    f"\n\nCURRENT PARTS LIBRARY:\n{lp.stdout[:5000]}")
        return f"unknown tool: {name}"
    return execute


def _commit(stage, slot, state_path, brief, project_stem=None) -> tuple[bool, dict]:
    sf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(slot, sf)
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


def drive_stage(client, stage, brief, state_path, workspace, max_tokens=4096, max_retries=2) -> dict:
    prep = _run(KICRAFT + ["stage-prep", stage, str(state_path)])
    if prep.returncode != 0:
        return {"stage": stage, "commit_ok": False, "cost_usd": 0.0,
                "error": f"stage-prep failed: {prep.stderr.strip()}"}
    prep_json = json.loads(prep.stdout)
    extras = prep_json.get("extras") or {}

    user = (f"PROJECT BRIEF:\n{brief}\n\n"
            f"CURRENT DESIGN STATE (JSON):\n{json.dumps(prep_json['state'])}")
    if extras:
        user += f"\n\nSTAGE EXTRAS (reference data from stage-prep):\n{json.dumps(extras)[:24000]}"
    user += f"\n\nProduce the {stage} slot JSON now."

    messages = [{"role": "system", "content": build_system(stage)},
                {"role": "user", "content": user}]
    tools = BOM_TOOLS if stage == "bom" else None
    executor = _bom_executor(workspace) if stage == "bom" else None

    total_cost = 0.0
    last: dict = {}
    for attempt in range(max_retries + 1):
        if tools:
            r = client.chat_with_tools(messages, tools, executor, max_tokens=max_tokens)
            raw, rounds = r["text"], r.get("rounds")
            total_cost += r["cost_usd"]
        else:
            res = client.chat(messages, max_tokens=max_tokens)
            raw, rounds = (res["text"] or res.get("reasoning") or ""), None
            total_cost += res["cost_usd"]
            messages.append({"role": "assistant", "content": raw})

        try:
            obj = _extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            last = {"error": "no JSON in reply", "reply_head": (raw or "")[:200], "rounds": rounds}
            messages.append({"role": "user", "content":
                             "That was not a single valid JSON object. Output ONLY the slot JSON."})
            continue

        project_stem = obj.pop("project_stem", None)
        ok, out = _commit(stage, dict(obj), state_path, brief, project_stem)
        if ok:
            return {"stage": stage, "commit_ok": True, "cost_usd": total_cost,
                    "attempts": attempt + 1, "rounds": rounds, "commit": out, "slot": obj}
        last = {"commit": out}
        msg = (f"stage-commit rejected that with errors: {json.dumps(out.get('errors'))}")
        if out.get("offenders"):
            msg += f"  offenders: {json.dumps(out.get('offenders'))}"
        msg += (". Fix exactly those (use the tools to resolve real symbols/footprints if a "
                "footprint did not resolve) and output ONLY the corrected slot JSON.")
        messages.append({"role": "user", "content": msg})

    return {"stage": stage, "commit_ok": False, "cost_usd": total_cost,
            "attempts": max_retries + 1, **last}


def drive_chain(stages, brief, workspace, max_tokens=4096, max_retries=2):
    ws = Path(workspace)
    (ws / ".kicraft").mkdir(parents=True, exist_ok=True)
    state_path = ws / ".kicraft" / "state.json"
    client = CappedOpenRouterClient(Settings.from_env())
    results = []
    for stage in stages:
        r = drive_stage(client, stage, brief, state_path, ws, max_tokens, max_retries)
        results.append(r)
        cost = r.get("cost_usd")
        cstr = f"${cost:.6f}" if isinstance(cost, (int, float)) else "n/a"
        tag = "ok  " if r.get("commit_ok") else "FAIL"
        extra = f" rounds={r['rounds']}" if r.get("rounds") else ""
        line = f"  [{tag}] {stage:<16} cost={cstr}  attempts={r.get('attempts', '-')}{extra}"
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
