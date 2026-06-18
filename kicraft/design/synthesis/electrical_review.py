"""LLM electrical-review pass (Layer 3 of the semantic-miswire reduction).

The deterministic §9 checks (incl. the Layer 1/2 semantic gates §9.16-§9.20)
prove the netlist is well-formed: pins exist, are covered, agree with net
polarity, aren't shorted, and obey known part-family pin roles. They cannot
judge whether the CIRCUIT is correct -- a 2nd-order filter tuned with 1st-order
math, a malformed R-2R ladder, an undersized bulk cap on a high-transient rail,
an MCU with no first-flash path. Those defects are electrically LEGAL (ERC/DRC
clean) but functionally wrong.

This module asks a capable model to review the committed design for exactly that
class, using ONLY structured design data: the intent, the BOM, and the netlist
rendered with pin FUNCTION NAMES. It deliberately never sees schematic geometry
-- ad-hoc geometry reading is what made the Claude-Code self-eval judges
hallucinate "pin scrambles" that the exported netlist showed were correct.

The client is injected (same contract as the eval judge):
``chat(messages, model=, max_tokens=, temperature=, meta_ctx=) ->
{"text": str, "cost_usd": float}``. This module therefore never imports the
server, and a fake client makes it testable offline.

Fail-closed: on malformed model output it retries once with the concrete defect,
then returns ``ok=False`` with no findings rather than inventing a verdict.
"""
from __future__ import annotations

import json

_SEVERITIES = ("blocker", "warning", "note")

_SYSTEM = (
    "You are a meticulous, skeptical hardware design reviewer. You review a "
    "COMPLETED KiCraft PCB design for ELECTRICAL CORRECTNESS using ONLY the "
    "structured digest provided (intent, BOM, and the netlist with pin function "
    "names). ERC/DRC-clean is NOT the same as correct: a well-formed netlist can "
    "still be a broken circuit. Judge the actual circuit: filter/feedback/divider "
    "math and component values, ladder/topology structure, decoupling and bulk "
    "capacitance sizing, an MCU's first-flash programming path, protection on "
    "exposed inputs, regulator thermal/current headroom, and whether the design "
    "honors the stated intent. Do NOT invent connections that are not in the "
    "digest, and do NOT speculate about physical layout/geometry -- you only have "
    "the netlist. Where the digest lacks evidence for a checklist item, treat it "
    "as NOT done. Report only concrete, defensible findings. Respond with a "
    "single JSON object and no other text."
)

_OUTPUT_CONTRACT = (
    "Return ONLY this JSON object (no markdown, no prose):\n"
    "{\n"
    '  "findings": [\n'
    '    {"severity": "blocker|warning|note", "area": "<short tag, e.g. '
    "'decoupling', 'filter-math', 'programming', 'protection'>\", "
    '"issue": "<what is wrong, citing the parts/nets>", '
    '"suggestion": "<the concrete fix>"}\n'
    "  ]\n"
    "}\n"
    "severity: 'blocker' = the board will not perform its core function or risks "
    "damage; 'warning' = a real weakness a reviewer would flag; 'note' = a "
    "nitpick. If the design is sound, return an empty findings list. Cite real "
    "refdes/net names from the digest in every issue."
)


# --------------------------------------------------------------------------- #
# digest
# --------------------------------------------------------------------------- #
def _pin_names(symbol, project_root):
    """{pin_number: pin_name} for a symbol, or {} if it can't be resolved."""
    from .symbol_pinout import SymbolNotFoundError, lookup_pins

    try:
        info = lookup_pins(symbol, project_root=project_root) if project_root else lookup_pins(symbol)
    except (SymbolNotFoundError, ValueError, TypeError):
        return {}
    return {p["number"]: (p.get("name") or "") for p in info["pins"]}


def build_design_digest(state, *, project_root=None, budget: int = 14000) -> str:
    """A compact, structured, geometry-free digest for the reviewer.

    Renders intent + architecture + BOM + the netlist with pin FUNCTION NAMES so
    the model reasons about what each pin does, never about pin numbers or
    coordinates.
    """
    parts: list[str] = []

    intent = state.intent
    if intent is not None:
        lines = [f"GOAL: {intent.goal}"]
        if intent.constraints:
            lines.append("CONSTRAINTS:\n" + "\n".join(f"  - {c}" for c in intent.constraints))
        if intent.named_parts:
            lines.append("NAMED PARTS: " + ", ".join(intent.named_parts))
        if intent.assumptions:
            lines.append("INTENT ASSUMPTIONS:\n" + "\n".join(f"  - {a}" for a in intent.assumptions))
        parts.append("INTENT (what the user asked for):\n" + "\n".join(lines))

    fs = state.functional_spec
    if fs is not None:
        try:
            blocks = ", ".join(b.name for b in fs.blocks)
        except AttributeError:
            blocks = ""
        if blocks:
            parts.append("FUNCTIONAL BLOCKS: " + blocks)

    arch = state.architecture
    if arch is not None:
        a = []
        a.append("SHEETS: " + ", ".join(s.name for s in arch.sheets))
        if arch.power_nets:
            a.append("POWER NETS: " + ", ".join(arch.power_nets))
        if arch.inter_sheet_nets:
            a.append("INTER-SHEET NETS: " + ", ".join(n.name for n in arch.inter_sheet_nets))
        parts.append("ARCHITECTURE:\n" + "\n".join(a))

    bom = state.bom
    if bom is not None:
        # parts table
        rows = []
        for p in bom.parts:
            mpn = getattr(p, "mpn", None) or getattr(p, "lcsc", None) or ""
            rows.append(f"  {p.ref:<6} {p.value:<16} {p.symbol}"
                        + (f"  [{mpn}]" if mpn else "")
                        + (f"  sheet={p.sheet}" if getattr(p, 'sheet', None) else ""))
        parts.append(f"BOM PARTS ({len(bom.parts)}):\n" + "\n".join(rows))

        # netlist with pin function names
        names_by_ref = {p.ref: _pin_names(p.symbol, project_root) for p in bom.parts}
        net_lines = []
        for c in bom.connections:
            eps = []
            for ep in c.endpoints:
                fn = names_by_ref.get(ep.ref, {}).get(ep.pin)
                eps.append(f"{ep.ref}.{ep.pin}" + (f"({fn})" if fn else ""))
            net_lines.append(f"  {c.net_name}: " + ", ".join(eps))
        if bom.no_connect_pins:
            nc = ", ".join(
                f"{ep.ref}.{ep.pin}" + (f"({names_by_ref.get(ep.ref, {}).get(ep.pin)})"
                                        if names_by_ref.get(ep.ref, {}).get(ep.pin) else "")
                for ep in bom.no_connect_pins
            )
            net_lines.append(f"  (no-connect: {nc})")
        parts.append(f"NETLIST (net: pins, with pin function names):\n" + "\n".join(net_lines))

    if state.open_questions:
        parts.append("OPEN QUESTIONS (already surfaced):\n"
                     + "\n".join(f"  - {q.text}" for q in state.open_questions))

    digest = "\n\n".join(parts)
    return digest[:budget]


# --------------------------------------------------------------------------- #
# review
# --------------------------------------------------------------------------- #
def _build_messages(digest: str) -> list[dict]:
    user = (
        f"{'=' * 60}\nDESIGN DIGEST (the only evidence; review strictly from it):\n"
        f"{'=' * 60}\n{digest}\n\n{_OUTPUT_CONTRACT}"
    )
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user}]


def _extract_json(text: str):
    """Parse the first JSON object out of a model reply, tolerating code fences."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t[3:]
        if t[:4].lower() == "json":
            t = t[4:]
        if "```" in t:
            t = t[:t.rfind("```")]
        t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _validate(obj):
    """(ok, findings, error). Each finding -> {severity, area, issue, suggestion}."""
    if not isinstance(obj, dict):
        return False, [], "no JSON object found in reply"
    raw = obj.get("findings")
    if not isinstance(raw, list):
        return False, [], "missing 'findings' array"
    out = []
    for i, f in enumerate(raw):
        if not isinstance(f, dict):
            return False, [], f"finding {i} is not an object"
        sev = str(f.get("severity", "")).lower()
        if sev not in _SEVERITIES:
            return False, [], f"finding {i} severity {f.get('severity')!r} not in {_SEVERITIES}"
        issue = f.get("issue")
        if not issue:
            return False, [], f"finding {i} missing 'issue'"
        out.append({
            "severity": sev,
            "area": str(f.get("area") or "").strip(),
            "issue": str(issue).strip(),
            "suggestion": str(f.get("suggestion") or "").strip(),
        })
    return True, out, None


def review_design(client, digest: str, *, model: str | None = None,
                  max_tokens: int = 3000, temperature: float = 0.0,
                  max_attempts: int = 2, reasoning: dict | None = None) -> dict:
    """Run the electrical review against a design digest.

    ``reasoning`` is the optional OpenRouter thinking-budget control (e.g.
    ``{"max_tokens": 8000}``) -- the review is a one-shot reasoning task, so a
    higher budget on a cheap model buys more than it costs. ``max_tokens`` covers
    the JSON answer and is set generously so reasoning never crowds it out.

    Returns ``{ok, findings, cost_usd, error, raw}``. ``findings`` is a list of
    ``{severity, area, issue, suggestion}`` (empty when the design is sound).
    Fails closed (``ok=False``, empty findings) after ``max_attempts`` rather
    than inventing a verdict.
    """
    messages = _build_messages(digest)
    total_cost = 0.0
    last_text = ""
    error = None
    for attempt in range(max_attempts):
        res = client.chat(messages, model=model, max_tokens=max_tokens,
                          temperature=temperature, reasoning=reasoning,
                          meta_ctx={"phase": "electrical_review", "stage": "review",
                                    "attempt": attempt})
        last_text = res.get("text") or ""
        total_cost += float(res.get("cost_usd") or 0.0)
        ok, findings, error = _validate(_extract_json(last_text))
        if ok:
            return {"ok": True, "findings": findings, "cost_usd": total_cost,
                    "error": None, "raw": last_text}
        messages.append({"role": "assistant", "content": last_text})
        messages.append({"role": "user", "content":
                         f"That response was not acceptable: {error}. Return ONLY the JSON "
                         "object with a 'findings' array; each finding needs a severity "
                         "(blocker|warning|note), area, issue, and suggestion."})

    return {"ok": False, "findings": [], "cost_usd": total_cost,
            "error": error or "review produced no valid verdict", "raw": last_text}


def has_blocker(findings) -> bool:
    return any(f.get("severity") == "blocker" for f in findings)
