"""Replace & rewire reconcile: LLM stacking headers -> standard headers."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from kicraft.design.models import BOM, BomPart, FormFactor, NetConnection, PinEndpoint
from kicraft.form_factors.reconcile import enforce_enabled, reconcile_standard_form_factor


def _shield_bom():
    parts = [
        BomPart(ref="U1", value="ME6211", symbol="me6211c33:ME6211C33M5G-N",
                footprint="me6211c33:SOT-23-5", sheet="REGULATOR"),
        BomPart(ref="C1", value="1uF", symbol="Device:C",
                footprint="Capacitor_SMD:C_0603_1608Metric", sheet="REGULATOR"),
        # Two LLM stacking headers on the interface sheet (to be replaced).
        BomPart(ref="J1", value="PinSocket_1x08", symbol="Connector_Generic:Conn_01x08",
                footprint="Connector_PinSocket_2.54mm:PinSocket_1x08_P2.54mm_Vertical",
                sheet="INTERFACE"),
        BomPart(ref="J2", value="PinHeader_1x08", symbol="Connector_Generic:Conn_01x08",
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x08_P2.54mm_Vertical",
                sheet="INTERFACE"),
    ]
    conns = [
        NetConnection(net_name="+5V", sheet="REGULATOR",
                      endpoints=[PinEndpoint(ref="U1", pin="5"), PinEndpoint(ref="J1", pin="1")]),
        NetConnection(net_name="GND", sheet="REGULATOR",
                      endpoints=[PinEndpoint(ref="U1", pin="2"), PinEndpoint(ref="J1", pin="4")]),
        # A signal net that ONLY involves a dropped header -> should vanish.
        NetConnection(net_name="D2_SENSE", sheet="INTERFACE",
                      endpoints=[PinEndpoint(ref="J2", pin="3")]),
    ]
    return BOM(parts=parts, connections=conns)


def _state(bom, standard="arduino_uno_shield"):
    ff = FormFactor(shape="rect", standard=standard) if standard else None
    return SimpleNamespace(intent=SimpleNamespace(form_factor=ff), bom=bom)


class TestReconcile:
    def test_replaces_stacking_headers_with_standard_headers(self):
        bom = _shield_bom()
        notes = reconcile_standard_form_factor(_state(bom))
        assert notes  # did something
        refs = {p.ref for p in bom.parts}
        assert "U1" in refs and "C1" in refs                # non-connectors kept
        # Exactly 4 headers now, and all are standard (identity via marker, not
        # ref name -- the freed J-refs are recycled J1..J4).
        headers = [p for p in bom.parts if p.symbol.startswith("Connector_Generic:Conn_")]
        assert len(headers) == 4
        assert all("standard form factor" in (p.sourcing_note or "") for p in headers)
        # No leftover LLM (unmarked) stacking header.
        assert not [p for p in headers if not (p.sourcing_note or "")]

    def test_power_rebinds_and_dangling_signal_net_dropped(self):
        bom = _shield_bom()
        reconcile_standard_form_factor(_state(bom))
        nets = {c.net_name for c in bom.connections}
        assert "+5V" in nets and "GND" in nets   # rails still present (rebound)
        assert "D2_SENSE" not in nets            # only referenced dropped J2 -> gone
        # Every connection endpoint references a part that still exists (no dangling).
        part_refs = {p.ref for p in bom.parts}
        assert all(ep.ref in part_refs for c in bom.connections for ep in c.endpoints)

    def test_signal_pins_marked_no_connect(self):
        bom = _shield_bom()
        reconcile_standard_form_factor(_state(bom))
        assert len(bom.no_connect_pins) == 22  # D0..D13 + A0..A5 + SCL + SDA

    def test_result_is_a_valid_bom(self):
        bom = _shield_bom()
        reconcile_standard_form_factor(_state(bom))
        # Re-validate the mutated BOM through the pydantic model (unique refs,
        # connections reference known parts, etc.).
        BOM.model_validate(bom.model_dump())

    def test_idempotent(self):
        bom = _shield_bom()
        reconcile_standard_form_factor(_state(bom))
        refs1 = sorted(p.ref for p in bom.parts)
        std1 = sorted(p.ref for p in bom.parts if "standard form factor" in (p.sourcing_note or ""))
        reconcile_standard_form_factor(_state(bom))  # run again
        std2 = sorted(p.ref for p in bom.parts if "standard form factor" in (p.sourcing_note or ""))
        # Standard headers are NOT re-dropped; count stays 4 (no churn/duplication).
        assert len(std2) == 4
        assert std1 == std2

    def test_noop_without_standard(self):
        bom = _shield_bom()
        before = len(bom.parts)
        assert reconcile_standard_form_factor(_state(bom, standard=None)) == []
        assert len(bom.parts) == before  # untouched

    def test_noop_for_unknown_standard(self):
        bom = _shield_bom()
        assert reconcile_standard_form_factor(_state(bom, standard="nope")) == []


class TestEnforceGate:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("KICRAFT_FORM_FACTOR_ENFORCE", raising=False)
        assert enforce_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "ON"])
    def test_on_values(self, monkeypatch, val):
        monkeypatch.setenv("KICRAFT_FORM_FACTOR_ENFORCE", val)
        assert enforce_enabled() is True

    def test_off_value(self, monkeypatch):
        monkeypatch.setenv("KICRAFT_FORM_FACTOR_ENFORCE", "0")
        assert enforce_enabled() is False
