"""Pin the electrical-review severity taxonomy to the human ground truth.

Asserts that the deterministic `_categorize` policy classifies every area in the
frozen bakeoff `labels.json` correctly: real (natural) blockers map to a
blocker-eligible category, and every expected warning caps at a warning ceiling.
This is what stops the policy from silently drifting away from the labels.

The bakeoff corpus lives under `logs/` which is gitignored, so these tests SKIP
(not fail) when the ground truth is not present (fresh clone / CI); they bite
locally and on the build host where the corpus exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.design.synthesis.electrical_review import _BLOCKER_ELIGIBLE, _categorize

_LABELS = (Path(__file__).resolve().parents[1]
           / "logs/bakeoff/20260618T200126Z/labels.json")


def _load():
    if not _LABELS.exists():
        pytest.skip(f"ground-truth labels not present: {_LABELS}")
    return json.loads(_LABELS.read_text())


def test_natural_blocker_areas_are_blocker_eligible():
    # Synthetics are EXCLUDED: labels._meta marks them calibration-floor only, and
    # syn_decap_drop's blocker area is `decoupling` (intentionally warning-max).
    labels = _load()
    for d in labels["designs"]:
        for b in d.get("true_blockers", []):
            cat = _categorize(b["area"])
            assert cat in _BLOCKER_ELIGIBLE, (d["design_id"], b["area"], cat)


def test_expected_warning_areas_cap_at_warning():
    labels = _load()
    for d in labels["designs"] + labels["synthetics"]:
        for w in d.get("expected_warnings", []):
            cat = _categorize(w["area"])
            assert cat not in _BLOCKER_ELIGIBLE, (d["design_id"], w["area"], cat)


def test_section9_synthetics_stay_blocker_eligible():
    # power-polarity / self-short / family-contract are caught deterministically by
    # section 9.16-9.20 AND remain blocker-eligible (defense in depth). decoupling
    # is calibration-only and intentionally demotes.
    labels = _load()
    syn = {d["design_id"]: d for d in labels["synthetics"]}
    for did in ("syn_vdd_gnd", "syn_self_short", "syn_can_miswire"):
        area = syn[did]["true_blockers"][0]["area"]
        assert _categorize(area) in _BLOCKER_ELIGIBLE, (did, area)
    assert _categorize("decoupling") not in _BLOCKER_ELIGIBLE
