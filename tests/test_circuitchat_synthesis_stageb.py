"""Stage-B end-to-end regression: the BMP280 reader must synthesize ERC-clean.

This is the canonical Stage-B fixture (``bom.connections`` populated), so it
exercises the comb-stub router, sheet placement, and root inter-sheet wiring —
the code paths the ``llups_like_state`` Stage-A fixture in
``test_circuitchat_synthesis.py`` never touches. It guards against the three
synthesis regressions found in the manual BMP280 test:

- RC1: pin coordinates snapped off their half-grid positions -> dangling wires;
- RC2: placement collisions -> two pins share one node (a short);
- RC3: the root sheet never wiring its inter-sheet pins.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from kicraft.circuitchat.models import ConversationState
from kicraft.circuitchat.synthesize import run

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bmp280_reader_state.json"

# `multiple_net_names` ("both X and Y are attached to the same items") is
# the reliable signal that two distinct nets were shorted onto one node.
# It is an ERC *warning*, so an errors-only check would miss a placement or
# routing collision -- assert on it explicitly to cover RC2. (We do NOT key
# on `pin_to_pin`: here it only flags vendored symbols whose power pins are
# typed "unspecified" touching a power symbol -- a symbol-quality nit, not a
# short.)
_SHORT_TYPES = {"multiple_net_names"}


@pytest.fixture
def bmp280_state() -> ConversationState:
    if not _FIXTURE.is_file():
        pytest.skip(f"fixture missing: {_FIXTURE}")
    return ConversationState.model_validate_json(_FIXTURE.read_text())


@pytest.mark.skipif(
    shutil.which("kicad-cli") is None, reason="kicad-cli not installed"
)
def test_bmp280_stageb_synthesizes_erc_clean(tmp_path, bmp280_state) -> None:
    """state.json -> synthesize -> KiCad ERC: zero errors and no shorts.

    ``run`` raises ``SynthesisValidationError`` if any check (incl. §9.12 ERC)
    fails, so reaching the assertions already means ERC errors == 0.
    """
    _artifacts, results = run(bmp280_state, tmp_path)

    erc = next((r for r in results if r.name.startswith("9.12")), None)
    assert erc is not None and erc.ok, (
        f"ERC check failed: {erc.message if erc else 'no ERC result'}"
    )

    report_path = tmp_path / f"{bmp280_state.project_stem}_erc.rpt"
    report = json.loads(report_path.read_text())
    shorts = [
        f"{s.get('path', '?')}: {v['type']} - {v.get('description', '')}"
        for s in report.get("sheets", [])
        for v in s.get("violations", [])
        if v.get("type") in _SHORT_TYPES
    ]
    assert not shorts, "unintended net shorts (placement collision?):\n" + "\n".join(
        shorts
    )
