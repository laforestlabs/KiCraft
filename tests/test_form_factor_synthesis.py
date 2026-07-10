"""Synthesis reconcile core: standard headers as real BOM parts, each pin bound
to a rail the design already carries or else no-connect (design-aware binding)."""

from __future__ import annotations

from kicraft.design.models import BomPart
from kicraft.form_factors import get_template
from kicraft.form_factors.synthesis import standard_form_factor_bom_delta


def _t():
    return get_template("arduino_uno_shield")


# The design's rails as an onboard-regulator shield actually names them: no `+`
# prefix, and the reserved/unused Arduino rails (VIN/IOREF/RESET/AREF) absent.
_DESIGN_RAILS = frozenset({"5V", "3V3", "GND"})


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


def test_binds_to_the_designs_own_rail_names_via_alias():
    # Header pins spelled +5V/+3V3 must land on the design's 5V/3V3 nets, not
    # introduce a second name for the same rail (which KiCad would merge).
    _, conns, _ = standard_form_factor_bom_delta(
        _t(), existing_refs=set(), design_rails=_DESIGN_RAILS
    )
    nets = {c.net_name for c in conns}
    assert nets == {"5V", "3V3", "GND"}
    assert "+5V" not in nets and "+3V3" not in nets


def test_unused_rails_and_signals_are_no_connect_not_bound():
    _, conns, ncs = standard_form_factor_bom_delta(
        _t(), existing_refs=set(), design_rails=_DESIGN_RAILS
    )
    bound_pins = {(ep.ref, ep.pin) for c in conns for ep in c.endpoints}
    nc_pins = {(ep.ref, ep.pin) for ep in ncs}
    # Every one of the 32 header pins is either bound or no-connect, never both.
    assert not (bound_pins & nc_pins)
    assert len(bound_pins) + len(ncs) == 32
    # A rail the design does NOT carry (VIN) is no-connect, not a dangling net.
    assert "VIN" not in {c.net_name for c in conns}


def test_no_design_rails_makes_every_pin_no_connect():
    _, conns, ncs = standard_form_factor_bom_delta(
        _t(), existing_refs=set(), design_rails=frozenset()
    )
    assert conns == []
    assert len(ncs) == 32  # nothing to bind onto -> all pins no-connect


def test_gnd_collects_all_pins_into_one_net():
    _, conns, _ = standard_form_factor_bom_delta(
        _t(), existing_refs=set(), design_rails=_DESIGN_RAILS
    )
    gnd = next(c for c in conns if c.net_name == "GND")
    # GND appears on digital_high (1) + power (2) = 3 header pins.
    assert len(gnd.endpoints) == 3


def test_connections_are_schema_valid_and_reserved_pin_excluded():
    parts, conns, ncs = standard_form_factor_bom_delta(
        _t(), existing_refs=set(), design_rails=_DESIGN_RAILS
    )
    part_refs = {p.ref for p in parts}
    for c in conns:
        assert len(c.endpoints) >= 1                     # NetConnection validator
        assert all(ep.ref in part_refs for ep in c.endpoints)
    # The reserved/NC power pin (power header, pin 1) must be no-connect, on no net.
    power_ref = parts[2].ref  # third header = power
    all_bound = {(ep.ref, ep.pin) for c in conns for ep in c.endpoints}
    all_nc = {(ep.ref, ep.pin) for ep in ncs}
    assert (power_ref, "1") not in all_bound
    assert (power_ref, "1") in all_nc
