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
import datetime as dt
import json
import os
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
# wiring is not a standalone slot model: it sets bom.connections + bom.no_connect_pins.
SUPPORTED_STAGES = (*SLOT_MODEL.keys(), "wiring")
# Full design order from a brief to a synthesizable state.
DESIGN_STAGES = ("intent", "functional_spec", "architecture", "bom", "wiring")

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
        "name": "search_symbols",
        "description": "Find the correct stock KiCad symbol id by keyword when you do not know "
                       "the exact 'Library:Name'. Returns matching 'Library:Name' ids to use "
                       "verbatim. Use this instead of guessing a symbol name.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "keywords, e.g. 'conn 02x08', 'crystal', "
                      "'n-channel mosfet'"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_footprints",
        "description": "Find the correct stock KiCad footprint id by keyword when you do not know "
                       "the exact 'Library:Name'. Returns matching 'Library:Name' ids to use "
                       "verbatim. Use this instead of guessing a footprint name.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "keywords, e.g. 'pinheader 2x08', "
                      "'barreljack', 'sot-23'"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "lookup_footprint",
        "description": "Verify a KiCad footprint in 'Library:Name' form exists and report its "
                       "pad count.",
        "parameters": {"type": "object", "properties": {
            "footprint": {"type": "string", "description": "e.g. "
                          "'Resistor_SMD:R_0603_1608Metric'"}}, "required": ["footprint"]}}},
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
            "footprint strings); search_symbols / search_footprints (find a stock KiCad symbol / "
            "footprint id by keyword); lookup_symbol (verify a 'Library:Name' symbol exists + "
            "pins); lookup_footprint (verify a footprint exists + pad count); lookup_lcsc_id "
            "(MPN/keyword -> LCSC C-number); add_part_from_lcsc (fetch a real symbol+footprint "
            "bundle into the project).\n"
            "- NEVER guess a stock 'Library:Name'. If unsure of the exact symbol OR footprint id, "
            "call search_symbols / search_footprints by keyword (e.g. 'conn 02x08', "
            "'pinheader 2x08', 'barreljack') to find it; a symbol or footprint that does not "
            "resolve is rejected at commit.\n"
            "- Use a library bundle VERBATIM when one matches (e.g. usb-c-16p for a USB-C "
            "receptacle): symbol '<name>:<sym>', footprint '<name>:<fp>'.\n"
            "- Trivial / generic parts from STOCK KiCad: discrete passives (R, C, L, LED, diode) "
            "AND generic mechanical/connectors (pin headers, barrel jacks, battery holders, basic "
            "switches). Use Device:R / Device:C / Device:L / Device:LED for passives. For their "
            "footprints use these DEFAULTS VERBATIM (already verified -- do NOT call a tool to "
            "check them): R -> Resistor_SMD:R_0805_2012Metric, C -> Capacitor_SMD:C_0805_2012Metric, "
            "L -> Inductor_SMD:L_0805_2012Metric, LED -> LED_SMD:LED_0805_2012Metric, "
            "diode -> Diode_SMD:D_SOD-123. Only call search_footprints for a passive if the board "
            "needs a DIFFERENT package (e.g. 0402, or through-hole 'LED_THT:LED_D5.0mm...'). For "
            "connectors/mechanical, call search_footprints for the exact id (e.g. "
            "'Connector_PinHeader_2.54mm:PinHeader_2x08_P2.54mm_Vertical').\n"
            "- ICs, sensors, MCUs, regulators, or ANY part where a specific MPN matters: do NOT "
            "pick a stock symbol/footprint. Resolve the real part: lookup_lcsc_id then "
            "add_part_from_lcsc, then list_parts to read the exact '<name>:<sym>' / '<name>:<fp>' "
            "strings. Substituting a generic stock part for a specific IC is wrong.\n"
            "- EFFICIENCY: when you need several independent lookups (e.g. several "
            "search_footprints, search_symbols, or lookup_lcsc_id for different parts), "
            "request them TOGETHER in a single turn (emit multiple tool calls at once) "
            "instead of one per turn. It is faster and far cheaper.\n"
            "- Every symbol AND footprint MUST resolve to a real file. When finished, output "
            "ONLY the BOM slot JSON.")
    if stage == "wiring":
        return (
            "\n- NET COVERAGE IS ENFORCED: every (ref, pin) of every part listed in "
            "extras.symbol_pinouts MUST appear either in a connections[].endpoints entry or "
            "in no_connect_pins. Omitting any pin fails the commit.\n"
            "- Use exact pin NUMBERS from extras.symbol_pinouts (never pin names).\n"
            "- Put genuinely-unused pins (USB-C SBU1/SBU2, shield, spare CC) in no_connect_pins.\n"
            "- net_name should match an architecture power_net or inter_sheet_net verbatim "
            "where applicable; connection.sheet must equal a bom part's sheet.")
    return ""


def _schema_for(stage: str) -> str:
    if stage == "wiring":
        return json.dumps({
            "type": "object",
            "properties": {
                "connections": {"type": "array",
                                "items": models.NetConnection.model_json_schema()},
                "no_connect_pins": {"type": "array",
                                    "items": models.PinEndpoint.model_json_schema()},
            },
            "required": ["connections", "no_connect_pins"],
        })
    return json.dumps(SLOT_MODEL[stage].model_json_schema())


def build_system(stage: str) -> str:
    spec = _spec_text(stage)
    schema = _schema_for(stage)
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
        '- Every "assumptions" entry must end with "(defaulted)".\n'
        "- CLARIFYING QUESTIONS: if the brief is too ambiguous to make a sound choice that "
        "materially changes the board, you MAY ask the user instead of guessing. To ask, "
        "output ONLY this shape (no slot this turn):\n"
        '  {"questions": [{"text": "...", "options": ["a suggested answer", "..."], '
        '"blocking": true}]}\n'
        "Ask at most 3 genuinely blocking questions, and only when a wrong guess would waste "
        "a real board; otherwise choose a sensible default, record it in \"assumptions\", and "
        "output the slot."
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
    # Tag part-query telemetry from the web path so part-query-report can split
    # hosted vs offline usage (query_log reads $KICRAFT_CALLER). Honors an
    # explicit override if the environment already set one.
    env = {**os.environ, "KICRAFT_CALLER": os.environ.get("KICRAFT_CALLER", "web")}
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
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
        if name == "search_symbols":
            r = _run(KICRAFT + ["search-symbols", str(args.get("query", ""))], workspace)
            return (r.stdout or r.stderr)[:3000]
        if name == "search_footprints":
            r = _run(KICRAFT + ["search-footprints", str(args.get("query", ""))], workspace)
            return (r.stdout or r.stderr)[:3000]
        if name == "lookup_footprint":
            r = _run(KICRAFT + ["lookup-footprint", str(args.get("footprint", ""))], workspace)
            return (r.stdout or r.stderr)[:3000]
        if name == "lookup_lcsc_id":
            r = _run(KICRAFT + ["lookup-lcsc-id", str(args.get("mpn", ""))], workspace)
            return (r.stdout or r.stderr)[:3000]
        if name == "add_part_from_lcsc":
            # Persist fetched parts to the shared HOME tier (not project): a part
            # the model needs once is then reused by every later design as a
            # `prototype`-badged bundle, so the catalog self-grows and repeated
            # LCSC fetches (the dominant BOM cost) amortize away.
            cmd = ["add-part", "--from-lcsc", str(args.get("lcsc_id", "")), "--into", "home"]
            if args.get("name"):
                cmd += ["--name", str(args["name"])]
            r = _run(KICRAFT + cmd, workspace)
            lp = _run(KICRAFT + ["list-parts"], workspace)
            return (f"add-part exit={r.returncode}\n{(r.stdout + chr(10) + r.stderr).strip()[:1500]}"
                    f"\n\nCURRENT PARTS LIBRARY:\n{lp.stdout[:5000]}")
        return f"unknown tool: {name}"
    return execute


def _commit(stage, slot, state_path, brief, project_stem=None, workspace=None) -> tuple[bool, dict]:
    sf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(slot, sf)
    sf.close()
    # Positionals (stage, state) BEFORE options: Python 3.12's argparse won't bind a
    # trailing optional positional that follows an option (e.g. --slot-file).
    cmd = KICRAFT + ["stage-commit", stage, str(state_path), "--slot-file", sf.name, "--no-archive"]
    if stage == "intent":
        cmd += ["--project-stem", project_stem or _fallback_stem(brief)]
    proc = _run(cmd, workspace)
    Path(sf.name).unlink(missing_ok=True)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        out = {"ok": False, "errors": [proc.stdout.strip() or proc.stderr.strip()]}
    return (proc.returncode == 0 and bool(out.get("ok"))), out


def _stamp_stage_status(state_path, stage: str, ok: bool, *,
                        cost_usd=None, attempts=None) -> None:
    """Record a stage's durable outcome in state.json's stage_status block (a real
    ConversationState field, so the CLI's load/validate/dump round-trip preserves
    it). This is what lets a reopened project restore its pipeline progress
    without the ephemeral event stream. Tolerates a missing state.json (a
    first-stage failure before any commit). Atomic write: the web render timer
    reads this file concurrently."""
    p = Path(state_path)
    try:
        sj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        sj = {}
    entry: dict = {"ok": bool(ok),
                   "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    if cost_usd is not None:
        entry["cost_usd"] = round(float(cost_usd), 6)
    if attempts is not None:
        entry["attempts"] = int(attempts)
    block = sj.get("stage_status") or {}
    block[stage] = entry
    sj["stage_status"] = block
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


# Per-stage self-correction budget. Wiring must satisfy whole-board net coverage
# (§9.11) in a single slot; on a complex board the model needs more correction
# passes than the simpler, smaller-slot stages, so they floor higher (BOM must
# also resolve every symbol/footprint to a real library entry within its budget).
_STAGE_MIN_RETRIES = {"wiring": 4, "bom": 4}


def _stage_max_retries(stage: str, default: int) -> int:
    return max(default, _STAGE_MIN_RETRIES.get(stage, 0))


# Tool-loop round budget for the BOM stage. The default (12) lets a weak model
# burn a dozen round-trips re-verifying a trivial 9-part BOM; 6 is plenty to
# resolve real parts, and client.chat_with_tools converges earlier when the
# model thrashes (identical-call cache + forced-final). Each stage attempt gets
# its own loop, so this is per-attempt.
_BOM_MAX_ROUNDS = 6


# Per-stage output token budget. Wiring emits the whole-board netlist in one slot;
# on a complex board that overflows the default cap and truncates into invalid
# JSON ("no JSON in reply"), so wiring floors higher.
_STAGE_MIN_TOKENS = {"wiring": 8192}


def _stage_max_tokens(stage: str, default: int) -> int:
    return max(default, _STAGE_MIN_TOKENS.get(stage, 0))


def _retry_feedback(out: dict) -> str:
    """Self-correction message fed back to the model after a rejected commit.

    Names the exact errors and offending pins, then instructs a *preserving patch*
    (keep every already-valid entry, change only the flagged ones) rather than a
    full redraft. On large list-shaped slots like wiring, re-emitting the whole slot
    each attempt tends to regress already-correct connections (whack-a-mole), so the
    preservation instruction is the lever that helps a weaker model converge.
    """
    msg = f"stage-commit rejected that with errors: {json.dumps(out.get('errors'))}"
    if out.get("offenders"):
        msg += f"  offenders: {json.dumps(out.get('offenders'))}"
    msg += (". Return the COMPLETE corrected slot JSON, preserving every entry that was "
            "already valid and changing ONLY the items listed above. When an offender lists "
            "'real options: ...', replace the bad id with ONE of those exact ids verbatim "
            "(do not invent or abbreviate); otherwise call search_symbols / search_footprints "
            "to find a real id. Do not drop or alter parts of the slot that were not flagged. "
            "Output ONLY the slot JSON.")
    return msg


def _normalize_questions(raw_list, stage: str) -> list[dict]:
    """Coerce a model-emitted questions payload into Question-shaped dicts (so the
    state.json open_questions list stays schema-valid). Caps count and lengths."""
    out = []
    for q in raw_list:
        if isinstance(q, dict) and str(q.get("text", "")).strip():
            out.append({
                "text": str(q["text"]).strip()[:500],
                "stage": stage,
                "blocking": bool(q.get("blocking", True)),
                "material": bool(q.get("material", True)),
                "options": [str(o)[:200] for o in (q.get("options") or [])][:6],
                "answer": None,
            })
    return out[:5]


def _attach_questions(state_path, stage: str, questions: list[dict]) -> None:
    """Write the stage's clarifying questions into state.json's open_questions
    (replacing any prior ones for this stage), so a reopened/parked project shows
    them. Tolerates a not-yet-created state.json (a first-stage question)."""
    try:
        sj = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        sj = {}
    kept = [q for q in (sj.get("open_questions") or []) if q.get("stage") != stage]
    sj["open_questions"] = kept + list(questions)
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(state_path).write_text(json.dumps(sj, indent=2) + "\n", encoding="utf-8")


def _client_model(client) -> str | None:
    """Best-effort display name of the model a client will call (shown in the UI)."""
    return getattr(getattr(client, "s", None), "model", None)


def drive_stage(client, stage, brief, state_path, workspace, max_tokens=4096, max_retries=2,
                progress=None, answers=None, instruction=None, meta_ctx=None) -> dict:
    if progress:
        progress({"kind": "stage_start", "stage": stage, "model": _client_model(client)})
    prep = _run(KICRAFT + ["stage-prep", stage, str(state_path)], workspace)
    if prep.returncode != 0:
        err = (prep.stderr.strip() or prep.stdout.strip())[:600]
        _stamp_stage_status(state_path, stage, False)
        if progress:
            progress({"kind": "stage_done", "stage": stage, "ok": False})
        return {"stage": stage, "commit_ok": False, "cost_usd": 0.0,
                "error": f"stage-prep failed: {err}"}
    prep_json = json.loads(prep.stdout)
    extras = prep_json.get("extras") or {}

    # Bookkeeping the model has no use for stays out of its prompt.
    prompt_state = dict(prep_json["state"])
    prompt_state.pop("stage_status", None)
    user = (f"PROJECT BRIEF:\n{brief}\n\n"
            f"CURRENT DESIGN STATE (JSON):\n{json.dumps(prompt_state)}")
    if extras:
        budget = 40000 if stage == "wiring" else 24000
        user += f"\n\nSTAGE EXTRAS (reference data from stage-prep):\n{json.dumps(extras)[:budget]}"
    if answers:
        qa = "\n".join(f"Q: {a.get('text', '')}\nA: {a.get('answer', '')}" for a in answers)
        user += f"\n\nThe user answered your earlier clarifying question(s):\n{qa}"
    if instruction:
        user += (f"\n\nThe user requests this change to the {stage}: {instruction}\n"
                 "Re-draft the slot to honor it, keeping everything else consistent.")
    user += f"\n\nProduce the {stage} slot JSON now."

    messages = [{"role": "system", "content": build_system(stage)},
                {"role": "user", "content": user}]
    tools = BOM_TOOLS if stage == "bom" else None
    executor = _bom_executor(workspace) if stage == "bom" else None

    total_cost = 0.0
    last: dict = {}
    cur_max_tokens = max_tokens
    for attempt in range(max_retries + 1):
        ctx = {**(meta_ctx or {}), "stage": stage, "attempt": attempt}
        tool_calls_ct = None
        if tools:
            r = client.chat_with_tools(messages, tools, executor, max_tokens=cur_max_tokens,
                                       max_rounds=_BOM_MAX_ROUNDS, progress=progress,
                                       meta_ctx=ctx)
            raw, rounds = r["text"], r.get("rounds")
            tool_calls_ct = r.get("tool_calls")
            finish = r.get("finish_reason")
            total_cost += r["cost_usd"]
        else:
            res = client.chat(messages, max_tokens=cur_max_tokens, progress=progress, meta_ctx=ctx)
            raw, rounds = (res["text"] or res.get("reasoning") or ""), None
            finish = res.get("finish_reason")
            total_cost += res["cost_usd"]
            messages.append({"role": "assistant", "content": raw})

        try:
            obj = _extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            last = {"error": "no JSON in reply", "reply_head": (raw or "")[:200],
                    "rounds": rounds, "tool_calls": tool_calls_ct}
            if finish == "length":
                # The reply hit the output cap and came back as truncated, invalid
                # JSON. A plain "try again" just truncates at the same spot, burning
                # another full-context call; give it more room for the next attempt.
                cur_max_tokens = min(cur_max_tokens * 2, 16384)
                messages.append({"role": "user", "content":
                                 "Your reply was cut off at the output token limit, so the "
                                 "JSON was truncated and invalid. The limit has been raised; "
                                 "output ONLY the slot JSON and keep it compact."})
            else:
                messages.append({"role": "user", "content":
                                 "That was not a single valid JSON object. Output ONLY the slot JSON."})
            continue

        # A clarifying-question payload parks the stage (no slot this turn). No slot
        # model has a top-level "questions" key, so the shape is unambiguous. Never
        # re-park right after an answer (caps the back-and-forth at one round/stage).
        qpayload = obj.get("questions") if isinstance(obj, dict) else None
        if isinstance(qpayload, list) and qpayload:
            qs = _normalize_questions(qpayload, stage)
            if any(q["blocking"] for q in qs) and not answers:
                _attach_questions(state_path, stage, qs)
                if progress:
                    progress({"kind": "question", "stage": stage, "questions": qs})
                return {"stage": stage, "commit_ok": False, "needs_input": True,
                        "questions": qs, "cost_usd": total_cost, "attempts": attempt + 1}
            messages.append({"role": "user", "content":
                             "Do not ask more questions. Apply sensible defaults (record each "
                             "in assumptions, ending '(defaulted)') and output ONLY the slot "
                             "JSON now."})
            continue

        project_stem = obj.pop("project_stem", None)
        ok, out = _commit(stage, dict(obj), state_path, brief, project_stem, workspace)
        if ok:
            _stamp_stage_status(state_path, stage, True,
                                cost_usd=total_cost, attempts=attempt + 1)
            if progress:
                progress({"kind": "stage_done", "stage": stage, "ok": True,
                          "cost": total_cost, "attempts": attempt + 1})
            return {"stage": stage, "commit_ok": True, "cost_usd": total_cost,
                    "attempts": attempt + 1, "rounds": rounds, "tool_calls": tool_calls_ct,
                    "commit": out, "slot": obj}
        last = {"commit": out}
        if progress:
            progress({"kind": "retry", "stage": stage, "errors": out.get("errors"),
                      "offenders": out.get("offenders")})
        messages.append({"role": "user", "content": _retry_feedback(out)})

    _stamp_stage_status(state_path, stage, False,
                        cost_usd=total_cost, attempts=max_retries + 1)
    if progress:
        progress({"kind": "stage_done", "stage": stage, "ok": False, "cost": total_cost})
    return {"stage": stage, "commit_ok": False, "cost_usd": total_cost,
            "attempts": max_retries + 1, "tool_calls": tool_calls_ct, **last}


def drive_chain(stages, brief, workspace, max_tokens=4096, max_retries=2, on_stage=None,
                progress=None, client=None, answers=None, instruction=None, run_id=None):
    ws = Path(workspace)
    (ws / ".kicraft").mkdir(parents=True, exist_ok=True)
    state_path = ws / ".kicraft" / "state.json"
    if client is None:
        client = CappedOpenRouterClient(Settings.from_env())
    base_ctx = {"run_id": run_id} if run_id else {}
    results = []
    for i, stage in enumerate(stages):
        # answers/instruction belong to the stage being resumed or edited, which
        # is the first stage of this chain; downstream stages re-draft cleanly.
        r = drive_stage(client, stage, brief, state_path, ws,
                        _stage_max_tokens(stage, max_tokens),
                        _stage_max_retries(stage, max_retries), progress=progress,
                        answers=(answers if i == 0 else None),
                        instruction=(instruction if i == 0 else None),
                        meta_ctx=base_ctx)
        results.append(r)
        if on_stage:
            on_stage(r)
        cost = r.get("cost_usd")
        cstr = f"${cost:.6f}" if isinstance(cost, (int, float)) else "n/a"
        tag = "ok  " if r.get("commit_ok") else "FAIL"
        extra = f" rounds={r['rounds']}" if r.get("rounds") else ""
        if r.get("tool_calls") is not None:
            extra += f" tools={r['tool_calls']}"
        line = f"  [{tag}] {stage:<16} cost={cstr}  attempts={r.get('attempts', '-')}{extra}"
        if not r.get("commit_ok"):
            line += f"\n         -> {r.get('error') or r.get('commit')}"
            if r.get("reply_head"):
                line += f"\n         reply_head: {r['reply_head']!r}"
        if r.get("needs_input"):
            line += "\n         -> parked: awaiting a clarifying answer from the user"
        print(line)
        if not r.get("commit_ok") or r.get("needs_input"):
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
    bad = [s for s in stages if s not in SUPPORTED_STAGES]
    if bad:
        ap.error(f"unsupported stage(s): {bad}; supported: {list(SUPPORTED_STAGES)}")

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
