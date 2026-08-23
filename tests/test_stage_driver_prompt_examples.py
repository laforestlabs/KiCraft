"""The worked examples embedded in the stage prompts MUST validate against the
real slot models (2026-07-19 review §7.1) — a schema change that breaks an
example must fail here, not teach the production model a guaranteed bounce.
"""
from __future__ import annotations

import json

from kicraft.design import models
from kicraft.server.stage_driver import _WORKED_EXAMPLES, build_system


def test_bom_example_validates_against_the_model():
    slot = json.loads(_WORKED_EXAMPLES["bom"])
    bom = models.BOM.model_validate(slot)
    assert [p.ref for p in bom.parts] == ["U1", "C1", "C2", "R1", "R2", "J1"]
    assert bom.ic_groups["U1"] == ["C1", "C2"]


def test_wiring_example_validates_as_bom_wiring_fields():
    slot = json.loads(_WORKED_EXAMPLES["wiring"])
    # Wiring commits into bom.connections/no_connect_pins; validate through
    # the BOM model carrying the same parts as the bom example.
    base = json.loads(_WORKED_EXAMPLES["bom"])
    base["connections"] = slot["connections"]
    base["no_connect_pins"] = slot["no_connect_pins"]
    bom = models.BOM.model_validate(base)
    nets = {c.net_name for c in bom.connections}
    assert nets == {"VIN", "+3V3", "GND", "NRST", "BOOT0"}


def test_examples_ride_the_system_prompt():
    assert _WORKED_EXAMPLES["bom"] in build_system("bom")
    assert _WORKED_EXAMPLES["wiring"] in build_system("wiring")
    assert "Worked example" not in build_system("intent")
