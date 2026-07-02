"""Suite-wide guards.

The live lcsc.com retail-stock check (``kicraft.parts_library.lcsc_retail``)
is consulted by the §9.26 BOM gate, ``lookup_lcsc_id`` and web pricing
whenever it is enabled — which is the default. Force it off for every test so
nothing hits the network implicitly (e.g. a stage-commit test running the
real gate against the host's real jlcparts dump). Tests that exercise retail
behavior install a fake module object (monkeypatching the importing module's
``lcsc_retail`` attribute) or re-enable via ``KICRAFT_LCSC_RETAIL=1``.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_live_retail_stock(monkeypatch):
    monkeypatch.setenv("KICRAFT_LCSC_RETAIL", "0")
