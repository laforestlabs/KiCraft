"""LLM-authored silkscreen content plan + deterministic corroboration lint.

The LLM authors CONTENT only (what the board should say); placement is pure
geometry in the build tail (``autoplacer/hardware/silk_legend.py``). This
module runs in the WEB process right after the post-wiring electrical review
(same client plumbing, same fail-soft rules) and commits the result to the
top-level ``state.silk_plan`` slot, so the no-LLM build worker and ``$0``
replays stay deterministic.

Trust model: a wrong voltage table printed on a board is worse than no
table, so every label passes ``lint_labels`` before commit — anchors must
name real BOM refs, and every numeric electrical claim (``9V``, ``3A`` ...)
must be corroborated by the design state (rail voltages, part values, the
brief/intent text). Uncorroborated labels are dropped WITH a recorded
reason (``SilkPlan.dropped_at_lint``) — a visible decision, never a silent
omission.
"""
from __future__ import annotations

import json
import re
import unicodedata

from kicraft.design.models import SilkAnchor, SilkLabel

from .electrical_review import _extract_json

_MAX_LABELS = 5
_MAX_LINES = 5
_MAX_LINE_CHARS = 30
_TITLE_MAX = 26

_SYSTEM = (
    "You write the silkscreen text for a small PCB, so the person HOLDING the "
    "physical board understands it without the schematic. From the design "
    "digest, author at most "
    f"{_MAX_LABELS} short labels: connector IO roles and ratings (what goes "
    "IN, what comes OUT, at which voltage/current), configuration "
    "switch/jumper tables (which setting selects what — derive the mapping "
    "from the digest's netlist and part values, e.g. CFG resistor ladders), "
    "and at most one critical usage note. STRICT EVIDENCE RULE: every claim "
    "must be derivable from the digest; if the digest does not establish a "
    "voltage/current/mapping, OMIT it rather than guess. Plain ASCII only. "
    "Keep every line under "
    f"{_MAX_LINE_CHARS} characters and every label under {_MAX_LINES} lines. "
    "Also produce a short human-readable board title (under "
    f"{_TITLE_MAX} characters). Respond with a single JSON object and no "
    "other text."
)

_OUTPUT_CONTRACT = (
    "Return ONLY this JSON object (no markdown, no prose):\n"
    "{\n"
    '  "title": "<short board title>",\n'
    '  "labels": [\n'
    '    {"id": "<kebab-slug>", "kind": "io|table|note", '
    '"text": "<line1\\nline2...>", '
    '"anchor": {"ref": "<refdes near which this belongs, e.g. J1>", '
    '"prefer": "above|below|left|right"}, "priority": 1}\n'
    "  ]\n"
    "}\n"
    "priority: 1 = must-have (the board is confusing without it), 2 = "
    "useful, 3 = nice to have. Anchor every label to the most relevant "
    "component (the connector it rates, the switch the table explains). "
    "For a DIP-switch table use one header line then one line per option, "
    "e.g. 'VOUT  1 2 3\\n 9V   ON - -\\n12V   - ON -\\n20V   - - ON'. "
    "An empty labels list is a valid answer for a board with nothing to "
    "explain."
)

# Electrical-claim tokens: "9V", "3.3 V", "500mA", "9/12/20V" (slash lists
# share the trailing unit). Case-insensitive on the unit, canonicalized.
_CLAIM_RE = re.compile(
    r"(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*)\s*(kV|mV|V|mA|uA|A|W|mAh|Ah)\b",
    re.IGNORECASE,
)
_UNIT_CANON = {"kv": "kV", "mv": "mV", "v": "V", "ma": "mA", "ua": "uA",
               "a": "A", "w": "W", "mah": "mAh", "ah": "Ah"}

_NON_ASCII = re.compile(r"[^\x20-\x7e\n]")
_ASCII_MAP = {"µ": "u", "μ": "u", "Ω": "ohm", "°": "deg", "±": "+/-",
              "·": "-", "×": "x", "—": "-", "–": "-"}

# Standalone small integers (switch-position indices) for the table guard.
_POSITION_RE = re.compile(r"(?<![\w.])([1-8])(?![\w.])")


def normalize_ascii(text: str) -> str:
    for k, v in _ASCII_MAP.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text)
    return _NON_ASCII.sub("", text)


def _claim_tokens(text: str) -> set[tuple[str, str]]:
    """{(number, unit)} electrical claims in ``text``, numbers normalized."""
    out: set[tuple[str, str]] = set()
    for numbers, unit in _CLAIM_RE.findall(text or ""):
        unit_c = _UNIT_CANON.get(unit.lower(), unit)
        for n in re.split(r"\s*/\s*", numbers):
            try:
                out.add((f"{float(n):g}", unit_c))
            except ValueError:
                continue
    return out


def build_corroboration_corpus(state) -> set[tuple[str, str]]:
    """Every electrical claim the design state can back.

    Sources: intent text, functional spec, architecture (incl. rail_voltages
    as explicit volt tokens), BOM values/notes/assumptions. A label claim
    outside this set is treated as hallucinated.
    """
    chunks: list[str] = []
    for slot in ("intent", "functional_spec", "architecture", "bom"):
        obj = getattr(state, slot, None)
        if obj is not None:
            try:
                chunks.append(obj.model_dump_json())
            except Exception:
                chunks.append(str(obj))
    corpus = _claim_tokens("\n".join(chunks))
    arch = getattr(state, "architecture", None)
    if arch is not None:
        rail_text = "\n".join(
            list((arch.rail_voltages or {}).keys()) + list(arch.power_nets or [])
        )
        corpus |= _claim_tokens(rail_text)
        for volts in (arch.rail_voltages or {}).values():
            try:
                corpus.add((f"{float(volts):g}", "V"))
            except (TypeError, ValueError):
                pass
    return corpus


def _switch_positions(part, project_root) -> int | None:
    """Position count for a switch-like part (pins // 2), or None."""
    try:
        from .symbol_pinout import lookup_pins

        info = lookup_pins(part.symbol, project_root=project_root)
        n_pins = len(info["pins"])
    except Exception:
        return None
    if n_pins < 2:
        return None
    return n_pins // 2


def lint_labels(
    raw_labels: list[dict], state, *, project_root=None
) -> tuple[list[SilkLabel], list[str]]:
    """Deterministic content lint. Returns (kept, dropped-reason-strings).

    Drops: unknown anchor refs, uncorroborated electrical claims, empty
    text, duplicate ids, DIP tables referencing more positions than the
    switch has. Trims to line/length caps. Order-preserving; caps the total
    at the highest-priority labels.
    """
    bom = getattr(state, "bom", None)
    parts_by_ref = {p.ref: p for p in (bom.parts if bom else [])}
    corpus = build_corroboration_corpus(state)

    kept: list[SilkLabel] = []
    dropped: list[str] = []
    seen_ids: set[str] = set()

    for i, raw in enumerate(raw_labels or []):
        if not isinstance(raw, dict):
            dropped.append(f"label[{i}]: not an object")
            continue
        label_id = str(raw.get("id") or f"label-{i}").strip() or f"label-{i}"
        if label_id in seen_ids:
            dropped.append(f"{label_id}: duplicate id")
            continue

        text = normalize_ascii(str(raw.get("text") or ""))
        lines = [ln.rstrip()[:_MAX_LINE_CHARS] for ln in text.split("\n")]
        lines = [ln for ln in lines if ln.strip()][:_MAX_LINES]
        text = "\n".join(lines)
        if not text.strip():
            dropped.append(f"{label_id}: empty text")
            continue

        anchor_raw = raw.get("anchor") or {}
        ref = str(anchor_raw.get("ref") or "").strip() or None
        if ref is not None and ref not in parts_by_ref:
            dropped.append(f"{label_id}: anchor {ref} not in BOM")
            continue
        prefer = anchor_raw.get("prefer")
        if prefer not in ("above", "below", "left", "right"):
            prefer = None

        claims = _claim_tokens(text)
        unbacked = sorted(f"{n}{u}" for n, u in claims - corpus)
        if unbacked:
            dropped.append(
                f"{label_id}: uncorroborated claim(s) {', '.join(unbacked)}"
            )
            continue

        kind = raw.get("kind") if raw.get("kind") in ("io", "table", "note") else "note"
        if kind == "table" and ref is not None:
            positions = _switch_positions(parts_by_ref[ref], project_root)
            if positions is not None:
                indices = [int(m) for m in _POSITION_RE.findall(lines[0])]
                if indices and max(indices) > positions:
                    dropped.append(
                        f"{label_id}: table names position {max(indices)} but "
                        f"{ref} has {positions}"
                    )
                    continue

        try:
            priority = min(3, max(1, int(raw.get("priority", 2))))
        except (TypeError, ValueError):
            priority = 2

        seen_ids.add(label_id)
        kept.append(SilkLabel(
            id=label_id, kind=kind, text=text,
            anchor=SilkAnchor(ref=ref, prefer=prefer) if ref else None,
            priority=priority,
        ))

    if len(kept) > _MAX_LABELS:
        by_priority = sorted(kept, key=lambda lb: lb.priority)
        cut = {id(lb) for lb in by_priority[_MAX_LABELS:]}
        for lb in kept:
            if id(lb) in cut:
                dropped.append(f"{lb.id}: over the {_MAX_LABELS}-label cap")
        kept = [lb for lb in kept if id(lb) not in cut]

    return kept, dropped


def author_labels(
    client, digest: str, *, model: str, reasoning=None,
    max_tokens: int = 1800, temperature: float = 0.2, max_attempts: int = 2,
) -> dict:
    """One LLM call (with a bounded retry on malformed JSON) -> raw plan.

    Returns {ok, title, labels, cost_usd, error}; ``labels`` are unlinted
    dicts — the caller runs :func:`lint_labels` before committing anything.
    """
    user = (
        f"{'=' * 60}\nDESIGN DIGEST (the only evidence; author strictly "
        f"from it):\n{'=' * 60}\n{digest}\n\n{_OUTPUT_CONTRACT}"
    )
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user}]
    total_cost = 0.0
    error = None
    for attempt in range(max_attempts):
        res = client.chat(messages, model=model, max_tokens=max_tokens,
                          temperature=temperature, reasoning=reasoning,
                          meta_ctx={"phase": "silk_plan", "stage": "silk_plan",
                                    "attempt": attempt})
        text = res.get("text") or ""
        total_cost += float(res.get("cost_usd") or 0.0)
        obj = _extract_json(text)
        if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
            title = normalize_ascii(str(obj.get("title") or "")).strip()
            return {"ok": True, "title": title[:_TITLE_MAX],
                    "labels": obj["labels"], "cost_usd": total_cost,
                    "error": None}
        error = "reply was not the required JSON object with a 'labels' array"
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content":
                         f"That response was not acceptable: {error}. "
                         "Return ONLY the JSON object."})
    return {"ok": False, "title": "", "labels": [], "cost_usd": total_cost,
            "error": error}


__all__ = [
    "author_labels",
    "lint_labels",
    "build_corroboration_corpus",
    "normalize_ascii",
]
