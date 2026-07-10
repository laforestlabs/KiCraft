"""Scaffold builders for replace & rewire (synthesis + compose data layers)."""

from __future__ import annotations

from kicraft.form_factors import get_template
from kicraft.form_factors.scaffold import (
    canonical_power_bindings,
    standard_header_parts,
    standard_placements,
)
from kicraft.form_factors.compose_scaffold import build_scaffold, resolve_scaffold


def _t():
    return get_template("arduino_uno_shield")


# --- synthesis data layer (scaffold.py) ---------------------------------------


class TestStandardHeaderParts:
    def test_emits_the_four_single_row_headers_as_bom_parts(self):
        parts = standard_header_parts(_t())
        assert [p["role"] for p in parts] == ["digital_high", "digital_low", "power", "analog"]
        assert [p["ref"] for p in parts] == ["J1", "J2", "J3", "J4"]
        # symbols track pin count
        by_role = {p["role"]: p for p in parts}
        assert by_role["digital_high"]["symbol"] == "Connector_Generic:Conn_01x10"
        assert by_role["analog"]["symbol"] == "Connector_Generic:Conn_01x06"
        # footprints are the template's stacking sockets
        assert "PinSocket_1x10" in by_role["digital_high"]["footprint"]

    def test_pins_carry_positions_and_nets_nc_is_none(self):
        parts = standard_header_parts(_t())
        power = next(p for p in parts if p["role"] == "power")
        assert power["pins"][0]["net"] is None  # pin 1 reserved/NC
        assert power["pins"][0]["x_mm"] == 27.94
        assert power["pins"][1]["net"] == "IOREF"

    def test_ref_start_offsets(self):
        parts = standard_header_parts(_t(), ref_start=7)
        assert [p["ref"] for p in parts] == ["J7", "J8", "J9", "J10"]


class TestPlacementsAndBindings:
    def test_placements_are_exact_and_locked(self):
        parts = standard_header_parts(_t())
        pl = standard_placements(parts)
        assert pl["J1"] == {"x_mm": 18.796, "y_mm": 2.54, "rotation_deg": 0.0, "locked": True}
        assert all(v["locked"] for v in pl.values())

    def test_only_canonical_rails_bind_and_gnd_lists_both_pins(self):
        parts = standard_header_parts(_t())
        b = canonical_power_bindings(parts)
        assert "+5V" in b and "+3V3" in b and "GND" in b and "VIN" in b
        # Signal pins are NOT auto-bound (wiring stage's job).
        assert "D0" not in b and "A0" not in b
        # The power header carries two GND pins.
        gnd_on_power = [rp for rp in b["GND"] if rp[0] == "J3"]
        assert len(gnd_on_power) == 2


# --- compose data layer (compose_scaffold.py) ---------------------------------


class TestComposeScaffold:
    def test_locked_connector_components_at_fixed_positions(self):
        s = build_scaffold(_t())
        assert set(s.components) == {"J1", "J2", "J3", "J4"}
        j1 = s.components["J1"]
        assert j1.locked is True
        assert (j1.pos.x, j1.pos.y) == (18.796, 2.54)
        assert len(j1.pads) == 10
        assert j1.kind == "connector"
        # NC pin -> empty net
        assert s.components["J3"].pads[0].net == ""

    def test_axis_x_headers_rotated_90_for_kicad_vertical_footprint(self):
        # A KiCad single-row vertical header advances +Y at rot 0; the Arduino
        # edge headers advance +X (axis="x"), so the stamp rotation is 90 deg so
        # the real footprint's pads land on the template pin coordinates.
        s = build_scaffold(_t())
        assert all(c.rotation == 90.0 for c in s.components.values())

    def test_real_refs_are_used_when_supplied(self):
        # The synthesis half's actual BOM refs (role -> ref) must be honored so
        # compose locks the SAME parts the schematic emitted.
        role_to_ref = {"digital_high": "J7", "digital_low": "J8",
                       "power": "J9", "analog": "J10"}
        s = build_scaffold(_t(), role_to_ref=role_to_ref)
        assert set(s.components) == {"J7", "J8", "J9", "J10"}
        # digital_high is the 10-pin header
        assert len(s.components["J7"].pads) == 10

    def test_outline_is_the_standard_rect(self):
        s = build_scaffold(_t())
        tl, br = s.outline
        assert (tl.x, tl.y) == (0.0, 0.0)
        assert (br.x, br.y) == (68.58, 53.34)

    def test_gate_off_by_default(self):
        assert resolve_scaffold({}) is None
        assert resolve_scaffold({"form_factor_standard": {"key": "arduino_uno_shield",
                                                          "validated": True}}) is None  # no enforce flag

    def test_gate_requires_validated(self):
        assert resolve_scaffold(
            {"form_factor_enforce": True,
             "form_factor_standard": {"key": "arduino_uno_shield", "validated": False}}
        ) is None

    def test_gate_on_returns_scaffold(self):
        s = resolve_scaffold(
            {"form_factor_enforce": True,
             "form_factor_standard": {"key": "arduino_uno_shield", "validated": True}}
        )
        assert s is not None and set(s.components) == {"J1", "J2", "J3", "J4"}

    def test_gate_honors_header_refs_from_cfg(self):
        s = resolve_scaffold(
            {"form_factor_enforce": True,
             "form_factor_standard": {
                 "key": "arduino_uno_shield", "validated": True,
                 "header_refs": {"digital_high": "J3", "digital_low": "J4",
                                 "power": "J5", "analog": "J6"}}}
        )
        assert s is not None and set(s.components) == {"J3", "J4", "J5", "J6"}

    def test_gate_unknown_key(self):
        assert resolve_scaffold(
            {"form_factor_enforce": True, "form_factor_standard": {"key": "nope", "validated": True}}
        ) is None
