"""Board-level features a brief states in words that name no part.

A prototyping area is the defining feature of a prototyping shield, and it is exactly the
kind of fact this pipeline loses: it names no component class, owns no pin and draws no
net, so the model may record it only as an adjective and every later stage has nothing to
build. This module is the deterministic net under that, in the same spirit as
:mod:`kicraft.design.synthesis.form_factor` for the board outline: the brief's own words
become a typed `fabrication` obligation at intent commit, and the realization stages derive
the sheet, the requirement and the pad field from it.

The obligation records the *fact*, never a size: a brief that asks for a prototyping area
rarely states one, and inventing a dimension here would be an unrequested constraint. The
default field geometry belongs to the lowering that draws it.
"""
from __future__ import annotations

import re

from kicraft.design.models import FabricationObligation

# The canonical feature name and the stable obligation id every stage reads.
PROTOTYPING_AREA_FEATURE = "prototyping-area"
PROTOTYPING_AREA_OBLIGATION_ID = "prototyping_area"

# `prototyping` (the gerund) or a named bare-board field. A bare "prototype" is NOT a
# match: "a prototype of a sensor board" asks for a one-off build, while "a prototyping
# shield/area" asks for a pad field the user will solder into.
_PROTOTYPING_AREA_RE = re.compile(
    r"\bprototyping\b"
    r"|\bproto[- _]?board\b"
    r"|\bperf(?:orated)?[- _]?board\b"
    r"|\bstrip[- _]?board\b"
    r"|\bpad[- _]?field\b",
    re.IGNORECASE,
)


def _feature_of(obligation) -> str:
    """The canonical fabrication feature of one obligation row, or ''."""
    kind = getattr(obligation, "kind", None)
    feature = getattr(obligation, "feature", None)
    if isinstance(obligation, dict):
        kind = obligation.get("kind")
        feature = obligation.get("feature")
    if kind != "fabrication":
        return ""
    return str(feature or "").strip().casefold().replace("_", "-")


def prototyping_area_requested(text: str) -> str | None:
    """The phrase in ``text`` that asks for a prototyping area, or None."""
    if not text:
        return None
    match = _PROTOTYPING_AREA_RE.search(text)
    return match.group(0) if match else None


def has_prototyping_area(obligations) -> bool:
    """Whether an obligation list already records the prototyping-area fact.

    Reads both dict rows (a stage candidate) and typed rows (committed state).
    """
    return any(_feature_of(row) == PROTOTYPING_AREA_FEATURE for row in obligations or ())


def extract_board_features(text: str) -> tuple[FabricationObligation, ...]:
    """Board-feature obligations a brief states, in canonical form.

    Empty when the text names no such feature. Never invents a size or a unit.
    """
    if prototyping_area_requested(text) is None:
        return ()
    return (
        FabricationObligation(
            kind="fabrication",
            original_obligation_id=PROTOTYPING_AREA_OBLIGATION_ID,
            feature=PROTOTYPING_AREA_FEATURE,
        ),
    )
