"""Replace & rewire reconcile: LLM stacking headers -> standard headers."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from kicraft.design.lowering import lower_requirement
from kicraft.design.models import (
    BOM,
    Architecture,
    BomPart,
    CircuitRequirement,
    FormFactor,
    InterSheetNet,
    NetConnection,
    PinEndpoint,
    Sheet,
    SheetPin,
)
from kicraft.form_factors import get_template
from kicraft.form_factors.scaffold import standard_header_parts
from kicraft.form_factors.reconcile import (
    _is_stacking_header,
    enforce_enabled,
    reconcile_standard_form_factor,
)


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
        # A requested signal remains available at the authoritative host pin.
        NetConnection(net_name="D2", sheet="INTERFACE",
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

    def test_detects_vendored_named_stacking_header(self):
        # WS5: the vendored library naming must be recognized -- keying only on the
        # KiCad-stock ``PinHeader_``/``P2.54mm`` substrings missed it.
        vendored = BomPart(
            ref="J1", value="Header 1x40", symbol="Connector_Generic:Conn_01x40",
            footprint="pin-header-female-2-54-1x40:HDR-TH_40P-P2.54-V-F", sheet="INTERFACE",
        )
        stock = BomPart(
            ref="J2", value="PinHeader_1x08", symbol="Connector_Generic:Conn_01x08",
            footprint="Connector_PinHeader_2.54mm:PinHeader_1x08_P2.54mm_Vertical", sheet="INTERFACE",
        )
        usb = BomPart(
            ref="J3", value="USB-C", symbol="usb:USB_C", footprint="usb:USB_C_Receptacle",
            sheet="INTERFACE",
        )
        assert _is_stacking_header(vendored)
        assert _is_stacking_header(stock)
        assert not _is_stacking_header(usb)  # a functional connector is left alone

    def test_replaces_vendored_named_stacking_headers(self):
        # proto-shield (Arduino Uno): the LLM headers use the vendored footprint,
        # so before the fix they were never dropped and the scaffold's J4 ref
        # collided with a leaf's parent-local twin at compose (WS5).
        parts = [
            BomPart(ref="U1", value="ATMEGA328", symbol="mcu:ATMEGA328",
                    footprint="mcu:TQFP-32", sheet="MCU"),
            BomPart(ref="J1", value="Header 1x40", symbol="Connector_Generic:Conn_01x40",
                    footprint="pin-header-female-2-54-1x40:HDR-TH_40P-P2.54-V-F", sheet="INTERFACE"),
            BomPart(ref="J3", value="Header 1x40", symbol="Connector_Generic:Conn_01x40",
                    footprint="pin-header-female-2-54-1x40:HDR-TH_40P-P2.54-V-F", sheet="INTERFACE"),
        ]
        conns = [
            NetConnection(net_name="+5V", sheet="MCU",
                          endpoints=[PinEndpoint(ref="U1", pin="1"), PinEndpoint(ref="J1", pin="1")]),
            NetConnection(net_name="GND", sheet="MCU",
                          endpoints=[PinEndpoint(ref="U1", pin="2"), PinEndpoint(ref="J1", pin="4")]),
        ]
        bom = BOM(parts=parts, connections=conns)
        # Must NOT raise the loud-survivor ValueError -- the headers are dropped.
        notes = reconcile_standard_form_factor(_state(bom))
        assert notes
        # Every remaining connector-class part is a standard-marked scaffold header.
        headers = [p for p in bom.parts if p.symbol.startswith("Connector_Generic:Conn_")]
        assert headers
        assert all("standard form factor" in (p.sourcing_note or "") for p in headers)

    def test_requested_signal_rebound_to_standard_pin(self):
        bom = _shield_bom()
        reconcile_standard_form_factor(_state(bom))
        low = next(p for p in bom.parts if (p.sourcing_note or "").endswith("digital_low"))
        endpoints = {
            (ep.ref, ep.pin) for c in bom.connections if c.net_name == "D2"
            for ep in c.endpoints
        }
        assert endpoints == {(low.ref, "6")}
        assert (low.ref, "6") not in {(ep.ref, ep.pin) for ep in bom.no_connect_pins}
        # Every connection endpoint references a part that still exists (no dangling).
        part_refs = {p.ref for p in bom.parts}
        assert all(ep.ref in part_refs for c in bom.connections for ep in c.endpoints)

    def test_unbound_pins_marked_no_connect(self):
        bom = _shield_bom()
        reconcile_standard_form_factor(_state(bom))
        # Only +5V/GND and the requested D2 signal are bound.
        assert len(bom.no_connect_pins) == 27
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
        # The physical interface still provides GND: migrate its sheet endpoint
        # rather than deleting the inter-sheet electrical contract.
        ground = next(net for net in arch.inter_sheet_nets if net.name == "GND")
        assert {ep.sheet for ep in ground.endpoints} == {"HOST HEADER", "REGULATOR"}

    def test_no_architecture_is_tolerated(self):
        bom = _shield_bom()
        # SimpleNamespace state with architecture=None must not raise.
        reconcile_standard_form_factor(_state(bom, architecture=None))


def _typed_proto_shield():
    """Oversized generic headers plus independent 20-pad prototype hardware."""
    template = get_template("arduino_uno_shield")
    geometry = standard_header_parts(template)
    parts = []
    requirements = []
    connections = []
    for index, header in enumerate(geometry, 1):
        # R13 assigned duplicate contacts on oversized generic connectors.
        nets = [pin["net"] or "NC" for pin in header["pins"]] * 2
        nets = ["3V3" if net == "+3V3" else "5V" if net == "+5V" else net for net in nets]
        requirement = CircuitRequirement(
            id=f"req_headers_{index}", sheet="STACKING HEADERS",
            role="connector", family="pin-header", parameters={"rows": 1},
            ports={f"pin{pin}": net for pin, net in enumerate(nets, 1)},
            functional_blocks=["STACKING_HEADERS"],
        )
        requirements.append(requirement)
        parts.append(BomPart(
            ref=f"J{index}", value=f"PinHeader_1x{len(nets):02d}",
            symbol=f"Connector_Generic:Conn_01x{len(nets):02d}",
            footprint=f"Connector_PinHeader_2.54mm:PinHeader_1x{len(nets):02d}_P2.54mm_Vertical",
            sheet=requirement.sheet, resolution_source="lowerer", resolution_id="pin-header@1",
            lowering_requirement_id=requirement.id, lowering_role="connector", lowering_index=0,
        ))
        for pin, net in enumerate(nets, 1):
            if net != "NC":
                connections.append(NetConnection(
                    net_name=net, sheet=requirement.sheet,
                    endpoints=[PinEndpoint(ref=f"J{index}", pin=str(pin))],
                ))
    prototype = CircuitRequirement(
        id="req_proto_area", sheet="PROTOTYPING AREA", role="connector",
        family="pin-header", parameters={"rows": 1},
        ports={f"pin{pin}": ("5V", "GND", "+3V3", "GND")[(pin - 1) % 4] for pin in range(1, 21)},
        functional_blocks=["PROTOTYPING_AREA"],
    )
    requirements.append(prototype)
    parts.append(BomPart(
        ref="J5", value="PinHeader_1x20", symbol="Connector_Generic:Conn_01x20",
        footprint="Connector_PinHeader_2.54mm:PinHeader_1x20_P2.54mm_Vertical",
        sheet=prototype.sheet, resolution_source="lowerer", resolution_id="pin-header@1",
        lowering_requirement_id=prototype.id, lowering_role="connector", lowering_index=0,
    ))
    for pin, net in prototype.ports.items():
        connections.append(NetConnection(
            net_name=net, sheet=prototype.sheet,
            endpoints=[PinEndpoint(ref="J5", pin=pin.removeprefix("pin"))],
        ))
    arch = Architecture(
        sheets=[Sheet(name=name, stem=name.replace(" ", "_"), function=name)
                for name in ("STACKING HEADERS", "PROTOTYPING AREA")],
        requirements=requirements, unresolved_requirement_ids=[r.id for r in requirements],
        power_nets=["5V", "GND", "3V3", "+3V3"], rail_voltages={"3V3": 3.3, "+3V3": 3.3},
        inter_sheet_nets=[InterSheetNet(name="+3V3", endpoints=[
            SheetPin(sheet=name, direction="bidirectional")
            for name in ("STACKING HEADERS", "PROTOTYPING AREA")
        ])],
    )
    return _state(BOM(parts=parts, connections=connections), architecture=arch)


class TestTypedInterfaceMigration:
    def test_preserves_prototype_hardware_signals_and_functional_owners(self):
        state = _typed_proto_shield()
        prototype = state.bom.parts[-1].model_dump()
        requirement = state.architecture.requirements[-1].model_dump()
        prototype_endpoints = {
            (c.net_name, ep.pin) for c in state.bom.connections for ep in c.endpoints if ep.ref == "J5"
        }
        owners = {p.ref: p.lowering_requirement_id for p in state.bom.parts}
        reconcile_standard_form_factor(state)
        assert next(p for p in state.bom.parts if p.ref == "J5").model_dump() == prototype
        assert next(r for r in state.architecture.requirements if r.id == "req_proto_area").model_dump() == requirement
        assert {p.ref: p.lowering_requirement_id for p in state.bom.parts} == owners
        assert {
            (c.net_name, ep.pin) for c in state.bom.connections for ep in c.endpoints if ep.ref == "J5"
        } == prototype_endpoints
        assert {s.name for s in state.architecture.sheets} == {"STACKING HEADERS", "PROTOTYPING AREA"}
        by_owner = {r.id: r for r in state.architecture.requirements}
        geometry = {
            header["role"]: header
            for header in standard_header_parts(get_template("arduino_uno_shield"))
        }
        for part in state.bom.parts:
            if part.ref == "J5":
                continue
            header = geometry[part.sourcing_note.rsplit(" ", 1)[1]]
            assert part.symbol == header["symbol"] and part.footprint == header["footprint"]
            artifact = lower_requirement(by_owner[part.lowering_requirement_id])
            assert artifact.groups[0].symbol == part.symbol
            assert artifact.groups[0].footprint == part.footprint
        endpoints = {
            (ep.ref, ep.pin): c.net_name for c in state.bom.connections for ep in c.endpoints
        }
        nc = {(ep.ref, ep.pin) for ep in state.bom.no_connect_pins}
        for part in state.bom.parts:
            for port, net in by_owner[part.lowering_requirement_id].ports.items():
                endpoint = (part.ref, port.removeprefix("pin"))
                if net == "NC":
                    assert endpoint in nc and endpoint not in endpoints
                else:
                    assert endpoints[endpoint] == net and endpoint not in nc
        assert set(endpoints.values()) >= {
            "5V", "+3V3", "GND", "VIN", "IOREF", "RESET", "AREF", "SCL", "SDA",
            *(f"D{index}" for index in range(14)), *(f"A{index}" for index in range(6)),
        }
        assert state.architecture.power_nets == ["5V", "GND", "+3V3"]
        assert state.architecture.rail_voltages == {"+3V3": 3.3}
        assert {
            net.name: {ep.sheet for ep in net.endpoints}
            for net in state.architecture.inter_sheet_nets
        } == {
            name: {"STACKING HEADERS", "PROTOTYPING AREA"}
            for name in ("5V", "+3V3", "GND")
        }
        BOM.model_validate(state.bom.model_dump())
        Architecture.model_validate(state.architecture.model_dump())

    def test_unsupported_signal_rejects_without_partial_migration(self):
        state = _typed_proto_shield()
        state.architecture.requirements[0].ports["pin1"] = "CUSTOM_SUPPLY"
        before = (state.bom.model_dump(), state.architecture.model_dump())
        with pytest.raises(ValueError, match="cannot preserve requested signal"):
            reconcile_standard_form_factor(state)
        assert (state.bom.model_dump(), state.architecture.model_dump()) == before

    def test_distinct_onboard_supplies_are_not_shorted_by_alias(self):
        state = _typed_proto_shield()
        state.bom.parts.append(BomPart(
            ref="U1", value="AMS1117-3.3", symbol="Regulator_Linear:AMS1117-3.3",
            footprint="Package_TO_SOT_SMD:SOT-223-3_TabPin2", sheet="PROTOTYPING AREA",
        ))
        state.bom.connections.append(NetConnection(
            net_name="3V3", sheet="PROTOTYPING AREA", endpoints=[PinEndpoint(ref="U1", pin="2")]
        ))
        before = (state.bom.model_dump(), state.architecture.model_dump())
        with pytest.raises(ValueError, match="distinct onboard rail bindings"):
            reconcile_standard_form_factor(state)
        assert (state.bom.model_dump(), state.architecture.model_dump()) == before


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
