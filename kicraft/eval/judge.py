"""Automated Class-J judge: the LLM "observer" for web self-evaluation.

In the offline harness a human observer grades the five judgment dimensions
(spec compliance, intent fidelity, electrical soundness, part selection, failure
honesty) against the rubric anchors. On the web there is no human in the loop, so
this module asks a capable model to do it: it renders the rubric's Class-J anchors
and observer-gate conditions verbatim, hands the model a compact run digest, and
requires a structured JSON verdict it can validate. The deterministic Class-C half
is untouched, so a weak or lenient judge can only move the judgment dimensions,
never the machine-measured ones.

The client is injected: any object exposing
``chat(messages, model=, max_tokens=, temperature=, meta_ctx=) -> {"text": str,
"cost_usd": float}`` works (the web app passes its capped OpenRouter client). This
module therefore never imports the server, and a fake client makes it testable
offline.

On malformed model output it retries once with the concrete defect, then fails
closed (levels left null, an error recorded) rather than guessing a score.
"""
from __future__ import annotations

import json
import re

_SYSTEM = (
    "You are a meticulous, skeptical hardware design reviewer. You grade a COMPLETED "
    "KiCraft PCB design run against a fixed rubric, using ONLY the evidence in the run "
    "digest provided. ERC-clean is not the same as correct: judge the actual circuit. "
    "Where the digest lacks evidence for a checklist item, treat it as NOT done rather "
    "than assuming the best. Do NOT compute or assert a specific numeric value (a "
    "voltage, current, resistance, reference voltage, or temperature) that is not given "
    "verbatim in the digest -- if a judgment requires a number the digest does not "
    "supply, say so in the evidence instead of estimating one. "
    "Respond with a single JSON object and no other text."
)


def _class_j_dims(rubric) -> list[dict]:
    return [d for d in rubric["dimensions"] if d.get("class") == "J"]


def _observer_gates(rubric) -> list[dict]:
    return [g for g in rubric["gates"] if g.get("detected_by") == "observer"]


def _render_rubric_section(jdims: list[dict], ogates: list[dict]) -> str:
    """Render the Class-J anchors + observer gates as plain text for the prompt."""
    out: list[str] = ["JUDGMENT DIMENSIONS (grade each on its 0-4 anchors):"]
    for d in jdims:
        out.append(f"\n### {d['id']}  (weight {d['weight']})")
        if d.get("summary"):
            out.append(d["summary"])
        if d.get("checklist"):
            out.append("Checklist:")
            out.extend(f"  - {c}" for c in d["checklist"])
        for key in ("examples", "examples_of_violation"):
            if d.get(key):
                out.append(f"{key.replace('_', ' ')}:")
                out.extend(f"  - {e}" for e in d[key])
        out.append("Anchors:")
        for level in sorted(d["anchors"]):
            out.append(f"  {level}: {d['anchors'][level]}")
    out.append(
        "\nOBSERVER GATES (include a gate in triggered_gates ONLY if it FIRES, "
        "with concrete evidence; do NOT enumerate gates that do not fire. If you "
        "mention a gate at all, set its \"triggered\" field explicitly):")
    for g in ogates:
        out.append(f"  - {g['id']} (cap {g['cap']}): {g['condition']}")
    return "\n".join(out)


def _output_contract(jdims: list[dict]) -> str:
    ids = ", ".join(f'"{d["id"]}"' for d in jdims)
    return (
        "Return ONLY this JSON object (no markdown, no prose):\n"
        "{\n"
        '  "dimensions": {\n'
        '    "<dimension_id>": {"level": <integer 0-4>, "evidence": "<one or two sentences '
        'citing the digest>"}\n'
        "  },\n"
        '  "triggered_gates": [ {"id": "<observer gate id>", "triggered": true, '
        '"evidence": "<why it holds>"} ]\n'
        "}\n"
        f"dimensions MUST contain exactly these keys: {ids}. "
        "Every level is an integer 0-4 per that dimension's anchors. "
        "triggered_gates may be empty; it is the list of gates that FIRE. An "
        'entry with "triggered": false is ignored -- prefer omitting non-firing '
        "gates entirely."
    )


def _build_messages(rubric_text: str, contract: str, digest: str) -> list[dict]:
    user = (
        f"{rubric_text}\n\n"
        f"{'=' * 60}\nRUN DIGEST (the only evidence; grade strictly from it):\n{'=' * 60}\n"
        f"{digest}\n\n"
        f"{contract}"
    )
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user}]


def _extract_json(text: str):
    """Parse the first JSON object out of a model reply, tolerating code fences and
    surrounding prose. Returns the object or None."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        # drop the opening fence (optionally ```json) and the closing fence
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


def _coerce_level(lvl):
    """Return an int 0-4 or None. Accepts 3 and 3.0; rejects bools and out-of-range."""
    if isinstance(lvl, bool):
        return None
    if isinstance(lvl, int) and 0 <= lvl <= 4:
        return lvl
    if isinstance(lvl, float) and lvl.is_integer() and 0 <= lvl <= 4:
        return int(lvl)
    return None


# Self-negating gate evidence: the judge sometimes ENUMERATES observer gates
# with a verdict in the evidence field instead of listing only firing ones
# (2026-07-27 batch: run_34 was capped 73->50 by evidence literally ending
# "Gate does not trigger."; run_17 capped 80.5->55 by "there is nothing to
# silently substitute against"). Applied only to legacy entries that carry no
# explicit "triggered" boolean; kept narrow so affirmative evidence that
# happens to contain "not surfaced"/"no open_question" is never screened.
_GATE_NEGATION_RE = re.compile(
    r"(?:\b(?:does|do|did)\s*(?:not|n't)\s+(?:trigger|hold|apply|fire)\b"
    r"|\bnot\s+triggered\b"
    r"|\bnothing\s+to\s+(?:\w+\s+){0,4}substitut"
    r"|\bno\s+(?:named|specific)\s+parts?\s+were\s+specified\b)",
    re.IGNORECASE,
)


def _validate(verdict, jdims: list[dict], ogates: list[dict]):
    """(ok, dims, gates, rejected, error). dims maps each J id ->
    {level, evidence}; ``gates`` are the AFFIRMED observer gates, ``rejected``
    the mentioned-but-not-firing ones (kept for the report, never applied)."""
    if not isinstance(verdict, dict):
        return False, {}, [], [], "no JSON object found in reply"
    dverd = verdict.get("dimensions")
    if not isinstance(dverd, dict):
        return False, {}, [], [], "missing 'dimensions' object"

    dims = {}
    for d in jdims:
        did = d["id"]
        entry = dverd.get(did)
        if not isinstance(entry, dict):
            return False, {}, [], [], f"dimension '{did}' missing or not an object"
        lvl = _coerce_level(entry.get("level"))
        if lvl is None:
            return False, {}, [], [], f"dimension '{did}' level not an integer 0-4 (got {entry.get('level')!r})"
        ev = entry.get("evidence")
        dims[did] = {"level": lvl, "evidence": str(ev) if ev is not None else ""}

    gate_caps = {g["id"]: g["cap"] for g in ogates}
    gates, rejected = [], []
    for g in (verdict.get("triggered_gates") or []):
        if not (isinstance(g, dict) and g.get("id") in gate_caps):
            continue
        why = str(g.get("evidence") or g.get("why") or "")
        rec = {"id": g["id"], "cap": gate_caps[g["id"]], "by": "observer",
               "why": why}
        trig = g.get("triggered")
        if trig is False:
            rec["rejected_because"] = "triggered: false"
            rejected.append(rec)
        elif trig is not True and _GATE_NEGATION_RE.search(why):
            # Legacy entry (no explicit boolean) whose evidence refutes itself.
            rec["rejected_because"] = "self-negating evidence"
            rejected.append(rec)
        else:
            gates.append(rec)
    return True, dims, gates, rejected, None


def grade_class_j(client, digest: str, rubric: dict, *, model: str | None = None,
                  max_tokens: int = 24000, temperature: float = 0.0,
                  max_attempts: int = 2) -> dict:
    """Grade the five Class-J dimensions and detect observer gates with an LLM.

    Returns ``{ok, dimensions, gates, gates_rejected, cost_usd, error, raw}``
    where ``dimensions`` maps each Class-J id to ``{level, evidence}`` (level is
    None on every dim when ``ok`` is False), ``gates`` is a list of AFFIRMED
    observer gates with caps, and ``gates_rejected`` the gates the judge
    mentioned but did not affirm (recorded, never applied). Fails closed after
    ``max_attempts`` rather than inventing a score.
    """
    jdims = _class_j_dims(rubric)
    ogates = _observer_gates(rubric)
    rubric_text = _render_rubric_section(jdims, ogates)
    contract = _output_contract(jdims)
    messages = _build_messages(rubric_text, contract, digest)

    total_cost = 0.0
    last_text = ""
    error = None
    for attempt in range(max_attempts):
        res = client.chat(messages, model=model, max_tokens=max_tokens,
                          temperature=temperature,
                          meta_ctx={"phase": "eval_judge", "stage": "judge", "attempt": attempt})
        last_text = res.get("text") or ""
        total_cost += float(res.get("cost_usd") or 0.0)
        ok, dims, gates, rejected, error = _validate(
            _extract_json(last_text), jdims, ogates)
        if ok:
            return {"ok": True, "dimensions": dims, "gates": gates,
                    "gates_rejected": rejected,
                    "cost_usd": total_cost, "error": None, "raw": last_text}
        # Repair: state the concrete defect and ask exactly once more.
        messages.append({"role": "assistant", "content": last_text})
        messages.append({"role": "user", "content":
                         f"That response was not acceptable: {error}. Return ONLY the JSON "
                         "object, with all five dimensions present, an integer level 0-4 and "
                         "an evidence string for each."})

    return {"ok": False,
            "dimensions": {d["id"]: {"level": None, "evidence": ""} for d in jdims},
            "gates": [], "gates_rejected": [], "cost_usd": total_cost,
            "error": error or "judge produced no valid verdict", "raw": last_text}
