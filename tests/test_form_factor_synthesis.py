"""Synthesis reconcile core: standard headers as real BOM parts + power nets."""

from __future__ import annotations

from kicraft.design.models import BomPart
from kicraft.form_factors import get_template
from kicraft.form_factors.synthesis import standard_form_factor_bom_delta


def _t():
    return get_template("arduino_uno_shield")


def test_emits_four_schema_valid_bom_parts():
    parts, _, _ = standard_form_factor_bom_delta(_t(), existing_refs=set())
    assert len(parts) == 4
    assert all(isinstance(p, BomPart) for p in parts)  # pydantic-validated
    roles = [p.value for p in parts]
    assert any("digital_high" in v for v in roles)
    assert all(p.sheet == "INTERFACE" for p in parts)


def test_refs_do_not_collide_with_existing():
    parts, _, _ = standard_form_factor_bom_delta(
        _t(), existing_refs={"J1", "J2", "U1", "C3"}
    )
    assert [p.ref for p in parts] == ["J3", "J4", "J5", "J6"]


def test_refs_start_at_j1_when_no_connectors():
    parts, _, _ = standard_form_factor_bom_delta(_t(), existing_refs={"U1", "C1"})
    assert [p.ref for p in parts] == ["J1", "J2", "J3", "J4"]


def test_power_nets_bound_signal_nets_not():
    _, conns, _ = standard_form_factor_bom_delta(_t(), existing_refs=set())
    nets = {c.net_name for c in conns}
    assert {"+5V", "+3V3", "GND", "VIN"} <= nets
    assert "D0" not in nets and "A0" not in nets  # wiring stage's job


def test_gnd_collects_all_pins_into_one_net():
    _, conns, _ = standard_form_factor_bom_delta(_t(), existing_refs=set())
    gnd = next(c for c in conns if c.net_name == "GND")
    # GND appears on digital_high (1) + power (2) = 3 header pins.
    assert len(gnd.endpoints) == 3


def test_connections_are_schema_valid_and_nc_excluded():
    parts, conns, _ = standard_form_factor_bom_delta(_t(), existing_refs=set())
    part_refs = {p.ref for p in parts}
    for c in conns:
        assert len(c.endpoints) >= 1                     # NetConnection validator
        assert all(ep.ref in part_refs for ep in c.endpoints)
    # NC pin (power pin 1) must not appear on any net.
    all_pins = {(ep.ref, ep.pin) for c in conns for ep in c.endpoints}
    power_ref = parts[2].ref  # third header = power
    assert (power_ref, "1") not in all_pins


def test_signal_pins_returned_as_noconnects():
    parts, conns, ncs = standard_form_factor_bom_delta(_t(), existing_refs=set())
    part_refs = {p.ref for p in parts}
    # D0..D13 (14) + A0..A5 (6) + SCL + SDA = 22 signal pins, all no-connect.
    assert len(ncs) == 22
    assert all(ep.ref in part_refs for ep in ncs)
    # None are power pins (those went to connections, not no-connect).
    bound = {(ep.ref, ep.pin) for c in conns for ep in c.endpoints}
    assert not ({(ep.ref, ep.pin) for ep in ncs} & bound)
