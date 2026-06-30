"""Deterministic outline-shape correctness check for the self-eval.

Given a built parent ``.kicad_pcb`` and the shape the brief requested, grade how
well the produced ``Edge.Cuts`` matches -- with no LLM, by classifying the
outline geometry (:mod:`kicraft.render.edge_cuts`) into a coarse family and
comparing it to the family the requested shape implies.

The grade is a 0-4 level (matching the eval rubric's Class-C anchor scale), so
``evaluate_outline_shape`` plugs straight into a future ``outline_shape_
correctness`` rubric dimension once the metrics dict carries the expected shape.
"""

from __future__ import annotations

from pathlib import Path

from kicraft.render.edge_cuts import classify_edge_cuts_shape, family_for_shape


def evaluate_outline_shape(pcb_path: Path, expected_shape: str | None) -> dict:
    """Grade the outline of ``pcb_path`` against ``expected_shape``.

    Returns ``{level, partial, rationale, expected_shape, expected_family,
    detected_family}``. ``level``:

    * **4** — outline family matches the requested shape's family.
    * **2** — non-rectangular, but a different family than requested (the shape
      "took" but isn't quite the one asked for).
    * **0** — rectangular when a non-rect shape was requested, or no Edge.Cuts.
    * ``None`` (partial) — no shape was requested (rectangular brief): not
      applicable, leave unscored.
    """
    expected = (expected_shape or "rect").strip().lower()
    expected_family = family_for_shape(expected)

    if expected in ("", "rect") or expected_family == "rectangular":
        return {
            "level": None, "partial": True,
            "rationale": "no non-rect shape requested; outline check N/A",
            "expected_shape": expected, "expected_family": "rectangular",
            "detected_family": None,
        }

    detected = classify_edge_cuts_shape(Path(pcb_path))
    if detected is None:
        return {
            "level": 0, "partial": False,
            "rationale": "no Edge.Cuts geometry to classify",
            "expected_shape": expected, "expected_family": expected_family,
            "detected_family": None,
        }

    det_family = detected["family"]
    if det_family == expected_family:
        level, why = 4, (
            f"outline family {det_family!r} matches requested {expected!r} "
            f"(label={detected['label']!r}, circularity={detected['circularity']})"
        )
    elif det_family != "rectangular":
        level, why = 2, (
            f"non-rectangular ({det_family!r}, label={detected['label']!r}) but "
            f"not the requested {expected_family!r} family"
        )
    else:
        level, why = 0, (
            f"requested {expected!r} ({expected_family!r}) but the board is "
            f"rectangular"
        )

    return {
        "level": level, "partial": False, "rationale": why,
        "expected_shape": expected, "expected_family": expected_family,
        "detected_family": det_family,
    }
