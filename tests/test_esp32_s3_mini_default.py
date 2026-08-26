"""ESP32-S3-MINI-1: vendored as the smaller native-USB default, with its antenna
keep-out wired into the placer config so the placer protects its PCB antenna.

The footprint-internal keep-out zone + courtyard are covered by
test_library_antenna_keepouts (the MINI footprint is discovered by name), and the
bundle loading clean by tests/parts_library/test_vendored_bundles_load. This file
guards the two MINI-specific links: the placer config glob matches the footprint,
and the architecture stage defaults to the MINI.
"""
from __future__ import annotations

import fnmatch

from kicraft.autoplacer.config import DEFAULT_CONFIG
from kicraft.server.stage_contracts import build_stage_response_contract
from kicraft.server.stage_prompts import build_system as _build_system


def build_system(stage: str) -> str:
    return _build_system(build_stage_response_contract(stage, {}))

_MINI_FOOTPRINT = "BULETM-SMD_ESP32-S3-MINI-1-N8"


def test_mini_footprint_matched_by_antenna_keepout_config():
    globs = DEFAULT_CONFIG["antenna_keepouts"]
    hits = [g for g in globs if fnmatch.fnmatch(_MINI_FOOTPRINT.lower(), g.lower())]
    assert hits, f"{_MINI_FOOTPRINT} matches no antenna_keepouts glob: {list(globs)}"
    rect = globs[hits[0]]
    # Antenna at the -y end: the near-field keep-out rect must sit in negative y,
    # spanning the body width across x.
    assert rect["y_min"] < rect["y_max"] <= 0
    assert rect["x_min"] < 0 < rect["x_max"]


def test_architecture_defaults_to_the_smaller_s3_mini():
    low = build_system("architecture").lower()
    assert "esp32-s3-mini-1" in low   # the smaller native-USB default
    assert "smaller" in low           # framed as the size-standardization default
    assert "esp32-s3-wroom-1" in low  # the GPIO step-up fallback is still named
