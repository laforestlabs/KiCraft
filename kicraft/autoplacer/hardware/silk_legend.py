"""Board legend + authored silk labels: payload build + subprocess driver.

The build tail calls :func:`apply_silk_legend` on the promoted routed board,
after routing/pour are final and BEFORE the fab-gate DRC (so the verify pass
sees the added silk). Content comes from ``state.silk_plan`` (authored by the
LLM pre-build and linted; see ``design/synthesis/silk_plan.py``); when the
slot is absent (old projects, replays, self-eval runs) the board still gets
the deterministic legend: title + "KiCraft [board-code] rev date".

Placement itself runs in ``_silk_legend_subprocess.py`` (pcbnew-in-subprocess
house rule) and is pure geometry: anything that does not fit is dropped and
reported, never overlapped onto pads or squeezed off the board.
"""
from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT = Path(__file__).parent / "_silk_legend_subprocess.py"

# Stroke-font silk stays ASCII: KiCad renders these fine, but gerber viewers
# and the stroke font are least surprising on plain ASCII (and the SSE
# mojibake incident taught us to normalize early).
_NON_ASCII = re.compile(r"[^\x20-\x7e\n]")

_TITLE_MAX = 32


def _ascii(text: str) -> str:
    import unicodedata

    replacements = {"µ": "u", "μ": "u", "Ω": "ohm", "°": "deg", "±": "+/-",
                    "·": "-", "×": "x", "—": "-", "–": "-"}
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text)
    return _NON_ASCII.sub("", text)


def _prettify_stem(stem: str) -> str:
    return re.sub(r"[_\s]+", " ", stem or "").strip()


def build_legend_lines(state, *, today: str | None = None) -> list[dict]:
    """The deterministic legend block: title line + attribution line.

    Works from whatever the state offers; every field degrades gracefully so
    id-less runs (self-eval, replay) still get "<stem> / KiCraft <date>".
    """
    plan = getattr(state, "silk_plan", None)
    title = (getattr(plan, "title", None) or "").strip()
    if not title:
        title = _prettify_stem(getattr(state, "project_stem", None) or "")
    title = _ascii(title)[:_TITLE_MAX].strip()

    board_code = (getattr(plan, "board_code", None) or "").strip()
    rev = (getattr(plan, "rev", None) or "1.0").strip()
    date = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    attribution = " ".join(
        p for p in ("KiCraft", board_code, f"rev {rev}", date) if p
    )

    lines: list[dict] = []
    if title:
        lines.append({"text": title, "height_mm": 1.2})
    lines.append({"text": _ascii(attribution), "height_mm": 0.8})
    return lines


def build_label_payload(state) -> list[dict]:
    """``state.silk_plan.labels`` -> placer label dicts (content pre-linted)."""
    plan = getattr(state, "silk_plan", None)
    if plan is None:
        return []
    out = []
    for lb in plan.labels:
        anchor = lb.anchor
        if lb.kind == "pinout":
            out.append({
                "id": lb.id,
                "kind": "pinout",
                "ref": getattr(anchor, "ref", None) if anchor else None,
                "pins": [{"pin": p.pin, "text": _ascii(p.text)} for p in lb.pins],
                "priority": lb.priority,
                "heights_mm": [0.8],
            })
            continue
        out.append({
            "id": lb.id,
            "text": _ascii(lb.text),
            "ref": getattr(anchor, "ref", None) if anchor else None,
            "prefer": getattr(anchor, "prefer", None) if anchor else None,
            "priority": lb.priority,
            "heights_mm": [1.0, 0.9, 0.8],
        })
    return out


def apply_silk_legend(pcb_path: Path, state, *, today: str | None = None) -> dict:
    """Stamp legend + labels onto ``pcb_path`` in place.

    Returns ``{"placed": [...], "dropped": [...]}`` from the placer. Raises
    on subprocess failure — the caller (build tail) treats silk as
    best-effort and must wrap this call.
    """
    from kicraft.autoplacer.routing_board import run_pcbnew_script_file

    payload = {
        "pcb_path": str(pcb_path),
        "output_path": str(pcb_path),
        "clearance_mm": 0.25,
        "edge_margin_mm": 0.5,
        "legend": {"lines": build_legend_lines(state, today=today), "gap_mm": 0.3},
        "labels": build_label_payload(state),
    }
    with tempfile.TemporaryDirectory(prefix="kicraft_silk_") as td:
        payload_path = Path(td) / "payload.json"
        result_path = Path(td) / "result.json"
        payload["result_path"] = str(result_path)
        payload_path.write_text(json.dumps(payload))
        run_pcbnew_script_file(str(_SCRIPT), str(payload_path))
        if not result_path.is_file():
            raise RuntimeError("silk legend subprocess wrote no result")
        return json.loads(result_path.read_text())


__all__ = ["apply_silk_legend", "build_legend_lines", "build_label_payload"]
