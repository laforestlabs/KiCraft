"""Replace & rewire reconcile: LLM stacking headers -> standard headers."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from kicraft.design.models import (
    BOM,
    Architecture,
    BomPart,
    FormFactor,
    InterSheetNet,
    NetConnection,
    PinEndpoint,
    Sheet,
    SheetPin,
)
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


def _state(bom, standard="arduino_uno_shield", architecture=None):
    ff = FormFactor(shape="rect", standard=standard) if standard else None
    return SimpleNamespace(
        intent=SimpleNamespace(form_factor=ff), bom=bom, architecture=architecture
    )


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

    def test_unbound_pins_marked_no_connect(self):
        bom = _shield_bom()
        reconcile_standard_form_factor(_state(bom))
        # The design carries only +5V and GND, so those header pins bind
        # (1 +5V + 3 GND = 4) and every other pin -- 3V3/VIN/IOREF/RESET/AREF,
        # all D/A/SCL/SDA, and the reserved power pin -- is no-connect (32-4=28).
        assert len(bom.no_connect_pins) == 28
        # A rail the design does NOT carry must not appear as a dangling net.
        nets = {c.net_name for c in bom.connections}
        assert "VIN" not in nets and "+3V3" not in nets and "AREF" not in nets

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


class TestEmptiedSheetPruning:
    """Consolidating the headers onto one host sheet empties any sheet that held
    only LLM connectors -- an empty sheet is a degenerate leaf that aborts the
    build, so the reconcile drops it from the architecture."""

    def _multi_sheet(self):
        # J1 on HOST HEADER (becomes the host sheet), J2/J3 on SPARE HEADER (both
        # dropped -> that sheet ends up empty), U1 on REGULATOR (keeps it alive).
        parts = [
            BomPart(ref="J1", value="PinHeader_1x08", symbol="Connector_Generic:Conn_01x08",
                    footprint="Connector_PinHeader_2.54mm:PinHeader_1x08_P2.54mm_Vertical", sheet="HOST HEADER"),
            BomPart(ref="J2", value="PinHeader_1x08", symbol="Connector_Generic:Conn_01x08",
                    footprint="Connector_PinHeader_2.54mm:PinHeader_1x08_P2.54mm_Vertical", sheet="SPARE HEADER"),
            BomPart(ref="J3", value="PinHeader_1x08", symbol="Connector_Generic:Conn_01x08",
                    footprint="Connector_PinHeader_2.54mm:PinHeader_1x08_P2.54mm_Vertical", sheet="SPARE HEADER"),
            BomPart(ref="U1", value="ME6211", symbol="me6211c33:ME6211C33M5G-N",
                    footprint="me6211c33:SOT-23-5", sheet="REGULATOR"),
        ]
        conns = [
            NetConnection(net_name="GND", sheet="REGULATOR",
                          endpoints=[PinEndpoint(ref="U1", pin="2"), PinEndpoint(ref="J1", pin="4")]),
        ]
        bom = BOM(parts=parts, connections=conns)
        arch = Architecture(
            sheets=[Sheet(name=n, stem=n.replace(" ", "_"), function=n)
                    for n in ("HOST HEADER", "SPARE HEADER", "REGULATOR")],
            power_nets=["GND"],
            inter_sheet_nets=[
                InterSheetNet(name="GND", endpoints=[
                    SheetPin(sheet="SPARE HEADER", direction="bidirectional"),
                    SheetPin(sheet="REGULATOR", direction="bidirectional"),
                ]),
            ],
        )
        return bom, arch

    def test_empty_sheet_dropped_from_architecture(self):
        bom, arch = self._multi_sheet()
        reconcile_standard_form_factor(_state(bom, architecture=arch))
        names = {s.name for s in arch.sheets}
        assert "SPARE HEADER" not in names             # emptied -> pruned
        assert {"HOST HEADER", "REGULATOR"} <= names    # host + regulator survive

    def test_inter_sheet_net_referencing_dropped_sheet_repaired(self):
        bom, arch = self._multi_sheet()
        reconcile_standard_form_factor(_state(bom, architecture=arch))
        # The GND inter-sheet net spanned SPARE HEADER<->REGULATOR; the spare sheet
        # is gone so it drops to one endpoint (no longer inter-sheet) and is
        # removed rather than left referencing a deleted sheet.
        for isn in arch.inter_sheet_nets:
            assert all(ep.sheet != "SPARE HEADER" for ep in isn.endpoints)

    def test_no_architecture_is_tolerated(self):
        bom = _shield_bom()
        # SimpleNamespace state with architecture=None must not raise.
        reconcile_standard_form_factor(_state(bom, architecture=None))


class TestEnforceGate:
    def test_default_on(self, monkeypatch):
        # The feature ships ON by default; unset (or empty) env => enabled.
        monkeypatch.delenv("KICRAFT_FORM_FACTOR_ENFORCE", raising=False)
        assert enforce_enabled() is True
        monkeypatch.setenv("KICRAFT_FORM_FACTOR_ENFORCE", "")
        assert enforce_enabled() is True

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "ON", "anything"])
    def test_on_values(self, monkeypatch, val):
        monkeypatch.setenv("KICRAFT_FORM_FACTOR_ENFORCE", val)
        assert enforce_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "OFF"])
    def test_kill_switch_values(self, monkeypatch, val):
        monkeypatch.setenv("KICRAFT_FORM_FACTOR_ENFORCE", val)
        assert enforce_enabled() is False
