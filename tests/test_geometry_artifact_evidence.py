"""Measured-geometry regressions for plausible physical false positives.

Every case delivers copper that resembles the required physical feature but is
not it: thermal support on another net, a header of the wrong physical size, an
absent or unrelated antenna clearance, and a touch electrode whose delivered
copper or underlay does not match the reviewed electrode.  Each case asserts the
fact the acceptance contract consumes, never the shape of the implementation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.design.part_identity import ReviewedPart, physical_inventory_record  # noqa: E402
from kicraft.eval import geometry_artifact_evidence as gae  # noqa: E402

_MM = pcbnew.FromMM

_POWER_FOOTPRINT = "aonr21357:DFN-8_L3.0-W3.0-P0.65-BL"
_POWER_SYMBOL = "aonr21357:AONR21357"
_HEADER_FOOTPRINT = "Connector_PinHeader_2.54mm:PinHeader_2x10_P2.54mm_Vertical"
_HEADER_SYMBOL = "Connector_Generic:Conn_02x10"
_ANTENNA_FOOTPRINT = "unictron-h2u38d1e1b0100:ANT-SMD_L3.2-W1.6-3"
_ANTENNA_RULE_AREA = "ANTENNA_CW324S_CLEARANCE"
_TOUCH_FOOTPRINT = "capacitive-touch-pad:TouchPad_12mm_Front_NoUnderlay"
_TOUCH_RULE_AREA = "touch_no_underlay"


def _board(path: Path, width: float = 40.0, height: float = 30.0):
    board = pcbnew.NewBoard(str(path))
    corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height), (0.0, 0.0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        segment = pcbnew.PCB_SHAPE(board)
        segment.SetShape(pcbnew.SHAPE_T_SEGMENT)
        segment.SetStart(pcbnew.VECTOR2I(_MM(x1), _MM(y1)))
        segment.SetEnd(pcbnew.VECTOR2I(_MM(x2), _MM(y2)))
        segment.SetLayer(pcbnew.Edge_Cuts)
        board.Add(segment)
    return board


def _nets(board, *names):
    for name in names:
        board.Add(pcbnew.NETINFO_ITEM(board, name))
    return lambda name: board.GetNetInfo().GetNetItem(name)


def _layers(*names):
    layers = pcbnew.LSET()
    for name in names:
        layers.AddLayer(getattr(pcbnew, name.replace(".", "_")))
    return layers


def _assign_net(item, netinfo) -> None:
    """Attach a net by code, after the item is owned by the board.

    ``BOARD.Add`` replaces a pre-assigned net, and ``SetNet`` leaves tracks and
    vias with a stale code on save, so both are avoided here.
    """
    item.SetNetCode(netinfo.GetNetCode())


def _footprint(board, ref: str, footprint: str, at: tuple[float, float]):
    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference(ref)
    library, _, item = footprint.partition(":")
    fp.SetFPID(pcbnew.LIB_ID(library, item))
    fp.SetPosition(pcbnew.VECTOR2I(_MM(at[0]), _MM(at[1])))
    board.Add(fp)
    return fp


def _smd_pad(fp, number: str, at: tuple[float, float], size: tuple[float, float], net=None, layers=("F.Cu",)):
    pad = pcbnew.PAD(fp)
    pad.SetNumber(number)
    pad.SetSize(pcbnew.VECTOR2I(_MM(size[0]), _MM(size[1])))
    pad.SetPosition(pcbnew.VECTOR2I(_MM(at[0]), _MM(at[1])))
    pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
    pad.SetLayerSet(_layers(*layers))
    fp.Add(pad)
    if net is not None:
        pad.SetNet(net)
    return pad


def _pth_pad(fp, number: str, at: tuple[float, float], net=None):
    pad = pcbnew.PAD(fp)
    pad.SetNumber(number)
    pad.SetSize(pcbnew.VECTOR2I(_MM(1.7), _MM(1.7)))
    pad.SetDrillSize(pcbnew.VECTOR2I(_MM(1.0), _MM(1.0)))
    pad.SetPosition(pcbnew.VECTOR2I(_MM(at[0]), _MM(at[1])))
    pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH)
    pad.SetLayerSet(pcbnew.PAD.PTHMask())
    fp.Add(pad)
    if net is not None:
        pad.SetNet(net)
    return pad


def _track(board, start: tuple[float, float], end: tuple[float, float], net, layer: str = "F.Cu"):
    track = pcbnew.PCB_TRACK(board)
    track.SetStart(pcbnew.VECTOR2I(_MM(start[0]), _MM(start[1])))
    track.SetEnd(pcbnew.VECTOR2I(_MM(end[0]), _MM(end[1])))
    track.SetWidth(_MM(0.25))
    track.SetLayer(getattr(pcbnew, layer.replace(".", "_")))
    board.Add(track)
    _assign_net(track, net)
    return track


def _via(board, at: tuple[float, float], net, width: float = 0.3):
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_MM(at[0]), _MM(at[1])))
    via.SetWidth(_MM(width))
    via.SetDrill(_MM(width / 2.0))
    board.Add(via)
    _assign_net(via, net)
    return via


def _plane(board, corners: list[tuple[float, float]], netinfo, layer: str = "F.Cu"):
    zone = pcbnew.ZONE(board)
    zone.SetIsRuleArea(False)
    zone.SetLayerSet(_layers(layer))
    zone.SetNetCode(netinfo.GetNetCode())
    outline = zone.Outline()
    outline.NewOutline()
    for x, y in corners:
        outline.Append(_MM(x), _MM(y))
    board.Add(zone)
    return zone


def _rule_area(fp, name: str, corners: list[tuple[float, float]], layers=("F.Cu",)):
    zone = pcbnew.ZONE(fp)
    zone.SetIsRuleArea(True)
    zone.SetZoneName(name)
    zone.SetDoNotAllowTracks(True)
    zone.SetDoNotAllowVias(True)
    zone.SetDoNotAllowPads(True)
    zone.SetDoNotAllowCopperPour(True)
    zone.SetDoNotAllowFootprints(True)
    zone.SetLayerSet(_layers(*layers))
    outline = zone.Outline()
    outline.NewOutline()
    for x, y in corners:
        outline.Append(_MM(x), _MM(y))
    fp.Add(zone)
    return zone


def _state(*rows):
    return {"bom": {"parts": [dict(row) for row in rows]}}


def _save_load(path: Path, board):
    """Deliver the board through a real KiCad save/load round trip."""
    board.Save(str(path))
    return pcbnew.LoadBoard(str(path))


def _facts(path: Path, board, state, slug: str):
    return gae.extract_geometry_facts(path.parent, state, _save_load(path, board), {"slug": slug})


@pytest.fixture
def reviewed_records(monkeypatch):
    """Resolve exact reviewed pairs like the pipeline, plus test-only records."""

    def install(*records: ReviewedPart):
        by_identity = {record.identity: record for record in records}
        original = gae.physical_inventory_record

        def resolve(*, mpn=None, symbol=None, footprint=None, datasheet=None, sourcing_note=None):
            record = by_identity.get(str(mpn or "").strip().casefold())
            if record is not None:
                return record if (record.symbol == symbol and record.footprint == footprint) else None
            return original(mpn=mpn, symbol=symbol, footprint=footprint)

        monkeypatch.setattr(gae, "physical_inventory_record", resolve)

    return install


def _power_record(*, pad_identity: bool = True) -> ReviewedPart:
    return ReviewedPart(
        identity="aonr21357",
        family="p-channel-highside-mosfet",
        package="AOS 3.0 x 3.0 mm DFN-8 with exposed drain thermal pad",
        bundle="aonr21357",
        symbol=_POWER_SYMBOL,
        footprint=_POWER_FOOTPRINT,
        physical_features=frozenset({"p-channel-mosfet", "highside-switch", "thermal-pad"}),
        contacts=tuple(str(number) for number in range(1, 10)),
        port_pins={"source": "1", "gate": "4", "drain": "5", "thermal_drain": "9"} if pad_identity else {},
        support_network={"thermal_pad": "9"} if pad_identity else {},
    )


def _power_board(
    path: Path,
    *,
    pad9_net: str = "DRAIN",
    pad7_net: str = "GND",
    pad7_vias: int = 3,
    pad9_vias: int = 0,
    drain_track: bool = False,
):
    """A power device whose exposed pad 9 is the drain, plus a GND pad 7.

    Stitching vias on pad 7 are the copper an old net-name heuristic mistook for
    the device's dissipation path; only pad 9's own net can carry it.
    """
    board = _board(path)
    net = _nets(board, "GND", "DRAIN")
    fp = _footprint(board, "Q1", _POWER_FOOTPRINT, (20.0, 15.0))
    _smd_pad(fp, "7", (14.0, 15.0), (2.4, 2.4), net(pad7_net))
    _smd_pad(fp, "9", (20.0, 15.0), (2.4, 1.7), net(pad9_net))
    for index in range(pad7_vias):
        _via(board, (13.6 + index * 0.5, 15.0), net(pad7_net))
    for index in range(pad9_vias):
        _via(board, (19.5 + index * 0.4, 15.0), net(pad9_net))
    if drain_track:
        _track(board, (20.0, 15.0), (24.0, 15.0), net(pad9_net))
    return board


def _power_state():
    return _state({"ref": "Q1", "mpn": "aonr21357", "symbol": _POWER_SYMBOL, "footprint": _POWER_FOOTPRINT})


def test_thermal_pad_ignores_vias_on_another_net(tmp_path, reviewed_records):
    # GND stitching vias on the device's GND pad are not dissipation for the
    # reviewed drain pad, which carries no same-net copper of its own.
    reviewed_records(_power_record())
    path = tmp_path / "b.kicad_pcb"
    board = _power_board(path)
    delivered = _save_load(path, board)
    assert {track.GetNetname() for track in delivered.GetTracks()} == {"GND"}
    facts = gae.extract_geometry_facts(path.parent, _power_state(), delivered, {"slug": "highside-switch-10a"})
    assert facts["gates"]["thermal_geometry"] == "fail"
    assert facts["geometry_diagnostics"]["thermal"]["reason"] == "no_same_net_thermal_copper"


def test_thermal_pad_accepts_same_net_vias(tmp_path, reviewed_records):
    reviewed_records(_power_record())
    path = tmp_path / "b.kicad_pcb"
    board = _power_board(path, pad9_vias=2)
    facts = _facts(path, board, _power_state(), "highside-switch-10a")
    assert facts["gates"]["thermal_geometry"] == "pass"


def test_thermal_via_required_only_where_the_brief_declares_it(tmp_path, reviewed_records):
    # A drain-side track is real thermal copper, so the brief that asks for the
    # declared feature passes; the brief that asks for thermal vias still fails.
    reviewed_records(_power_record())
    path = tmp_path / "b.kicad_pcb"
    board = _power_board(path, drain_track=True)
    highside = _facts(path, board, _power_state(), "highside-switch-10a")
    buck = _facts(path, board, _power_state(), "buck-3a")
    assert highside["gates"]["thermal_geometry"] == "pass"
    assert buck["gates"]["thermal_geometry"] == "fail"
    assert buck["geometry_diagnostics"]["thermal"]["reason"] == "no_same_net_thermal_via"


def test_thermal_pad_ignores_a_same_net_plane_elsewhere(tmp_path, reviewed_records):
    # Copper on the reviewed pad's net that is nowhere near it, and not joined to
    # it, is not dissipation at the pad.
    reviewed_records(_power_record())
    path = tmp_path / "b.kicad_pcb"
    board = _power_board(path, pad7_vias=0)
    _plane(board, [(30.0, 20.0), (38.0, 20.0), (38.0, 28.0), (30.0, 28.0)],
           board.GetNetInfo().GetNetItem("DRAIN"))
    facts = _facts(path, board, _power_state(), "highside-switch-10a")
    assert facts["gates"]["thermal_geometry"] == "fail"
    assert facts["geometry_diagnostics"]["thermal"]["reason"] == "no_same_net_thermal_copper"


def test_thermal_gate_is_unverified_without_a_reviewed_pad_identity(tmp_path, reviewed_records):
    # The old heuristic passed this board on a GND-named pad plus GND vias; with
    # no reviewed exposed-pad identity the honest fact is unverified, not pass.
    reviewed_records(_power_record(pad_identity=False))
    path = tmp_path / "b.kicad_pcb"
    board = _power_board(path, pad9_net="GND", pad7_vias=5)
    facts = _facts(path, board, _power_state(), "highside-switch-10a")
    assert "thermal_geometry" not in facts["gates"]
    assert facts["geometry_diagnostics"]["thermal"]["status"] == "unverified"


_USB_Mpn = "12401610e4-2a"
_USB_SYMBOL = "Connector:USB_C_Receptacle"
_USB_FOOTPRINT = "Connector_USB:USB_C_Receptacle_Amphenol_12401610E4-2A"


def _usb_record() -> ReviewedPart:
    """The real reviewed USB-C pair, resolved exactly as the pipeline does."""
    record = physical_inventory_record(mpn=_USB_Mpn, symbol=_USB_SYMBOL, footprint=_USB_FOOTPRINT)
    assert record is not None
    return record


def _devboard(path: Path, header_pads: list[tuple[float, float]]):
    board = _board(path, 60.0, 80.0)
    net = _nets(board, "GPIO", "VBUS")
    usb = _footprint(board, "J1", _USB_FOOTPRINT, (1.5, 40.0))
    _smd_pad(usb, "A1", (1.5, 40.0), (8.0, 7.0), net("VBUS"))
    header = _footprint(board, "J2", _HEADER_FOOTPRINT, (57.0, 40.0))
    for number, (x, y) in enumerate(header_pads, start=1):
        _pth_pad(header, str(number), (x, y), net("GPIO"))
    return board


def _devboard_state():
    usb = _usb_record()
    return _state(
        {"ref": "J1", "mpn": usb.identity, "symbol": usb.symbol, "footprint": usb.footprint},
        {"ref": "J2", "mpn": None, "symbol": _HEADER_SYMBOL, "footprint": _HEADER_FOOTPRINT},
    )


def test_opposite_edge_requires_a_two_by_ten_header(tmp_path):
    path = tmp_path / "b.kicad_pcb"
    pads = [(56.5 + 2.54 * column, 30.0 + 2.54 * row) for column in (0, 1) for row in range(10)]
    facts = _facts(path, _devboard(path, pads), _devboard_state(), "rounded-c3-devboard")
    assert facts["net_paths"] == {"usb_c_edge": True, "gpio_header_opposite_edge": True}
    assert facts["gates"]["connector_edges"] == "pass"


def test_opposite_edge_rejects_a_single_row_of_twenty_pins(tmp_path):
    path = tmp_path / "b.kicad_pcb"
    pads = [(59.04, 30.0 + 2.54 * row) for row in range(20)]
    facts = _facts(path, _devboard(path, pads), _devboard_state(), "rounded-c3-devboard")
    assert facts["net_paths"]["gpio_header_opposite_edge"] is False
    assert facts["gates"]["connector_edges"] == "fail"


def test_opposite_edge_rejects_a_two_by_ten_header_at_the_wrong_pitch(tmp_path):
    path = tmp_path / "b.kicad_pcb"
    pads = [(57.0 + 2.0 * column, 30.0 + 2.0 * row) for column in (0, 1) for row in range(10)]
    facts = _facts(path, _devboard(path, pads), _devboard_state(), "rounded-c3-devboard")
    assert facts["net_paths"]["gpio_header_opposite_edge"] is False
    assert facts["gates"]["connector_edges"] == "fail"


_ANTENNA_SHAPE = [
    {"x": -2.0, "y": 0.5}, {"x": 2.0, "y": 0.5}, {"x": 2.0, "y": 2.6}, {"x": -2.0, "y": 2.6},
]


def _antenna_record(*, shape=_ANTENNA_SHAPE) -> ReviewedPart:
    metadata = {
        "rule_area_name": _ANTENNA_RULE_AREA,
        "edge_required": True,
        "prohibited": ["tracks", "vias", "pads", "copperpour", "footprints"],
        "source": "Unictron H2U38D1E1B0100 datasheet ground-layout figure",
    }
    if shape is not None:
        metadata["local_shape"] = shape
    return ReviewedPart(
        identity="h2u38d1e1b0100",
        family="chip-antenna-2g4",
        package="Unictron CW324S ceramic chip antenna 3.2 x 1.6 x 1.3 mm",
        bundle="unictron-h2u38d1e1b0100",
        symbol="unictron-h2u38d1e1b0100:H2U38D1E1B0100",
        footprint=_ANTENNA_FOOTPRINT,
        physical_features=frozenset({"chip-antenna", "2.4ghz-antenna"}),
        contacts=("1", "2"),
        support_network={"rf_antenna_geometry": metadata},
    )


def _antenna_board(path: Path, *, rule_area: list[tuple[float, float]] | None, track: bool = False):
    board = _board(path)
    net = _nets(board, "RF_FEED")
    fp = _footprint(board, "AE1", _ANTENNA_FOOTPRINT, (20.0, 0.6))
    _smd_pad(fp, "1", (18.42, 0.6), (0.4, 0.4), net("RF_FEED"))
    _smd_pad(fp, "2", (21.58, 0.6), (0.4, 0.4))
    if track:
        _track(board, (19.0, 2.0), (21.0, 2.0), net("RF_FEED"))
    if rule_area is not None:
        _rule_area(fp, _ANTENNA_RULE_AREA, rule_area)
    return board


def _antenna_state():
    return _state({"ref": "AE1", "mpn": "h2u38d1e1b0100",
                   "symbol": "unictron-h2u38d1e1b0100:H2U38D1E1B0100", "footprint": _ANTENNA_FOOTPRINT})


def test_antenna_clearance_matching_the_reviewed_shape_passes(tmp_path, reviewed_records):
    reviewed_records(_antenna_record())
    path = tmp_path / "b.kicad_pcb"
    delivered = [(18.0, 1.1), (22.0, 1.1), (22.0, 3.2), (18.0, 3.2)]
    facts = _facts(path, _antenna_board(path, rule_area=delivered), _antenna_state(), "nrf52-beacon")
    assert facts["gates"]["rf_antenna_geometry"] == "pass"


def test_antenna_clearance_with_delivered_copper_inside_fails(tmp_path, reviewed_records):
    reviewed_records(_antenna_record())
    path = tmp_path / "b.kicad_pcb"
    delivered = [(18.0, 1.1), (22.0, 1.1), (22.0, 3.2), (18.0, 3.2)]
    board = _antenna_board(path, rule_area=delivered, track=True)
    facts = _facts(path, board, _antenna_state(), "nrf52-beacon")
    assert facts["gates"]["rf_antenna_geometry"] == "fail"
    assert facts["geometry_diagnostics"]["rf_antenna"]["reason"] == "delivered_copper_inside_antenna_clearance"


def test_antenna_clearance_unrelated_to_the_reviewed_shape_fails(tmp_path, reviewed_records):
    # A real rule area somewhere else is not the manufacturer's clearance.
    reviewed_records(_antenna_record())
    path = tmp_path / "b.kicad_pcb"
    delivered = [(24.0, 1.1), (28.0, 1.1), (28.0, 3.2), (24.0, 3.2)]
    facts = _facts(path, _antenna_board(path, rule_area=delivered), _antenna_state(), "nrf52-beacon")
    assert facts["gates"]["rf_antenna_geometry"] == "fail"
    assert facts["geometry_diagnostics"]["rf_antenna"]["reason"] == "delivered_antenna_clearance_mismatch"


def test_antenna_gate_is_unverified_without_a_source_backed_shape(tmp_path, reviewed_records):
    # A named area with prohibition flags is intent, not a measured clearance.
    reviewed_records(_antenna_record(shape=None))
    path = tmp_path / "b.kicad_pcb"
    delivered = [(18.0, 1.1), (22.0, 1.1), (22.0, 3.2), (18.0, 3.2)]
    facts = _facts(path, _antenna_board(path, rule_area=delivered), _antenna_state(), "nrf52-beacon")
    assert "rf_antenna_geometry" not in facts["gates"]
    assert facts["geometry_diagnostics"]["rf_antenna"]["status"] == "unverified"


def test_antenna_gate_fails_when_the_reviewed_clearance_is_absent(tmp_path, reviewed_records):
    reviewed_records(_antenna_record())
    path = tmp_path / "b.kicad_pcb"
    facts = _facts(path, _antenna_board(path, rule_area=None), _antenna_state(), "nrf52-beacon")
    assert facts["gates"]["rf_antenna_geometry"] == "fail"
    assert facts["geometry_diagnostics"]["rf_antenna"]["reason"] == "delivered_antenna_clearance_missing"


def _touch_record() -> ReviewedPart:
    return ReviewedPart(
        identity="pcb-fabricated-touch-electrode-12mm",
        family="capacitive-touch-pad",
        package="board-fabricated 12 mm round electrode",
        bundle="capacitive-touch-pad",
        symbol="capacitive-touch-pad:CapacitiveTouchPad",
        footprint=_TOUCH_FOOTPRINT,
        physical_features=frozenset({"capacitive-touch-pad"}),
        contacts=("1",),
        support_network={"touch_electrode": {
            "rule_area_name": _TOUCH_RULE_AREA,
            "source": "Microchip AN2934 Capacitive Touch Sensor Design Guide",
            "electrode_diameter_mm": 12.0,
            "underlay_clearance_mm": 2.0,
            "prohibited": ["tracks", "vias", "pads", "copperpour", "footprints"],
        }},
    )


def _touch_board(path: Path, *, pad_size=(12.0, 12.0), pad_layers=("F.Cu",), keepout=8.0, underlay=False):
    board = _board(path)
    net = _nets(board, "TOUCH1")
    fp = _footprint(board, "PAD1", _TOUCH_FOOTPRINT, (20.0, 15.0))
    _smd_pad(fp, "1", (20.0, 15.0), pad_size, net("TOUCH1"), layers=pad_layers)
    _rule_area(fp, _TOUCH_RULE_AREA,
               [(20.0 - keepout, 15.0 - keepout), (20.0 + keepout, 15.0 - keepout),
                (20.0 + keepout, 15.0 + keepout), (20.0 - keepout, 15.0 + keepout)],
               layers=("B.Cu",))
    if underlay:
        _track(board, (18.0, 15.0), (22.0, 15.0), net("TOUCH1"), layer="B.Cu")
    return board


def _touch_state():
    return _state({"ref": "PAD1", "mpn": "pcb-fabricated-touch-electrode-12mm",
                   "symbol": "capacitive-touch-pad:CapacitiveTouchPad", "footprint": _TOUCH_FOOTPRINT})


def test_touch_electrode_with_delivered_copper_and_clearance_passes(tmp_path, reviewed_records):
    reviewed_records(_touch_record())
    path = tmp_path / "b.kicad_pcb"
    facts = _facts(path, _touch_board(path), _touch_state(), "chamfered-badge")
    assert facts["gates"]["touch_no_underlay"] == "pass"


def test_touch_electrode_of_the_wrong_size_fails(tmp_path, reviewed_records):
    reviewed_records(_touch_record())
    path = tmp_path / "b.kicad_pcb"
    facts = _facts(path, _touch_board(path, pad_size=(6.0, 6.0)), _touch_state(), "chamfered-badge")
    assert facts["gates"]["touch_no_underlay"] == "fail"
    assert facts["geometry_diagnostics"]["touch_electrodes"]["reason"] == "delivered_electrode_geometry_mismatch"


def test_touch_electrode_left_as_bare_copper_fails(tmp_path, reviewed_records):
    reviewed_records(_touch_record())
    path = tmp_path / "b.kicad_pcb"
    board = _touch_board(path, pad_layers=("F.Cu", "F.Mask"))
    facts = _facts(path, board, _touch_state(), "chamfered-badge")
    assert facts["gates"]["touch_no_underlay"] == "fail"
    assert facts["geometry_diagnostics"]["touch_electrodes"]["reason"] == "delivered_electrode_bare_copper"


def test_touch_electrode_underlay_clearance_too_small_fails(tmp_path, reviewed_records):
    reviewed_records(_touch_record())
    path = tmp_path / "b.kicad_pcb"
    facts = _facts(path, _touch_board(path, keepout=6.5), _touch_state(), "chamfered-badge")
    assert facts["gates"]["touch_no_underlay"] == "fail"
    assert facts["geometry_diagnostics"]["touch_electrodes"]["reason"] == "delivered_electrode_underlay_too_small"


def test_touch_electrode_with_copper_routed_under_it_fails(tmp_path, reviewed_records):
    reviewed_records(_touch_record())
    path = tmp_path / "b.kicad_pcb"
    facts = _facts(path, _touch_board(path, underlay=True), _touch_state(), "chamfered-badge")
    assert facts["gates"]["touch_no_underlay"] == "fail"
    assert facts["geometry_diagnostics"]["touch_electrodes"]["reason"] == "delivered_copper_under_electrode"


# ------------------------- proto-shield prototyping area ---------------------
#
# The feature is the pad field itself: the field's own pads are obstacles to the
# free-area search, so a board that DELIVERS a 2.54 mm field can have almost no
# free area left and must still pass. The field is proved by copper (lone
# through-hole pads on one uniform grid), never by a label.

_PAD_FOOTPRINT = "prototyping-area:PrototypingPad_1.5mm_Drill0.8mm"


def _pad_field_board(
    path: Path,
    *,
    rows: int = 5,
    cols: int = 5,
    pitch: float = 2.54,
    width: float = 14.0,
    height: float = 14.0,
    origin: tuple[float, float] = (2.0, 2.0),
):
    board = _board(path, width, height)
    index = 1
    for row in range(rows):
        for column in range(cols):
            at = (origin[0] + column * pitch, origin[1] + row * pitch)
            fp = _footprint(board, f"PB{index}", _PAD_FOOTPRINT, at)
            _pth_pad(fp, "1", at)
            index += 1
    return board


def _pad_field_state(rows: int = 5, cols: int = 5):
    return _state(
        *(
            {
                "ref": f"PB{index}",
                "value": "Prototyping pad, 2.54 mm pitch",
                "symbol": "prototyping-area:PrototypingPad",
                "footprint": _PAD_FOOTPRINT,
            }
            for index in range(1, rows * cols + 1)
        )
    )


def _proto(path: Path, board, state, slug: str = "proto-shield"):
    facts = _facts(path, board, state, slug)
    return facts["gates"]["prototyping_area"], facts["geometry_diagnostics"]["prototyping_area"]


def test_delivered_pad_field_is_a_usable_prototyping_area(tmp_path):
    path = tmp_path / "b.kicad_pcb"
    gate, proto = _proto(path, _pad_field_board(path), _pad_field_state())
    assert gate == "pass"
    assert proto["pad_grid"]["usable"] is True
    assert proto["pad_grid"]["holes"] == 25
    assert (proto["pad_grid"]["columns"], proto["pad_grid"]["rows"]) == (5, 5)
    assert proto["pad_grid"]["pitch_mm"] == pytest.approx(2.54)
    # The field's own pads are obstacles, so this pass comes from the delivered grid.
    assert proto["free_area"]["usable"] is False
    assert proto["usable"] is True


def test_free_board_area_still_passes_without_a_pad_field(tmp_path):
    path = tmp_path / "b.kicad_pcb"
    gate, proto = _proto(path, _board(path, 40.0, 30.0), _state())
    assert gate == "pass"
    assert proto["free_area"]["usable"] is True
    assert proto["free_area"]["grid_holes"] >= 25
    assert proto["free_area"]["pitch_mm"] == pytest.approx(2.54)
    assert proto["pad_grid"]["holes"] == 0
    assert proto["usable"] is True


def test_a_pad_field_off_the_0_1_inch_grid_is_not_usable(tmp_path):
    # DIP parts and 0.1 inch headers fit 0.1 inch only, so a field spread at some
    # other spacing is not this feature, however many pads it holds.
    path = tmp_path / "b.kicad_pcb"
    gate, proto = _proto(
        path, _pad_field_board(path, pitch=3.1925, width=18.0, height=18.0), _pad_field_state()
    )
    assert proto["pad_grid"]["usable"] is False
    assert gate == "fail"


def test_pad_field_below_the_minimum_hole_count_is_not_usable(tmp_path):
    path = tmp_path / "b.kicad_pcb"
    gate, proto = _proto(
        path,
        _pad_field_board(path, rows=4, cols=5, width=12.0, height=12.0),
        _pad_field_state(4, 5),
    )
    assert proto["pad_grid"]["pads"] == 20
    assert proto["pad_grid"]["holes"] == 0
    assert proto["pad_grid"]["usable"] is False
    assert gate == "fail"


def test_scattered_through_hole_pads_are_not_a_pad_field(tmp_path):
    # 25 lone pads on a diagonal are not a grid, however many there are.
    path = tmp_path / "b.kicad_pcb"
    board = _board(path, 12.0, 12.0)
    for index in range(25):
        at = (1.0 + 0.4 * index, 1.0 + 0.4 * index)
        fp = _footprint(board, f"PB{index + 1}", _PAD_FOOTPRINT, at)
        _pth_pad(fp, "1", at)
    gate, proto = _proto(path, board, _pad_field_state())
    assert proto["pad_grid"]["pads"] == 25
    assert proto["pad_grid"]["holes"] == 0
    assert gate == "fail"


def test_a_pin_bank_is_not_a_pad_field(tmp_path):
    # A header's pads share one footprint; a pad field is lone plated holes.
    path = tmp_path / "b.kicad_pcb"
    board = _board(path, 14.0, 14.0)
    fp = _footprint(board, "J1", _HEADER_FOOTPRINT, (7.0, 7.0))
    for column in range(10):
        for row in range(10):
            _pth_pad(fp, str(column * 10 + row + 1), (2.0 + column * 1.0, 2.0 + row * 1.0))
    gate, proto = _proto(path, board, _state())
    assert proto["pad_grid"]["pads"] == 0
    assert proto["pad_grid"]["holes"] == 0
    assert gate == "fail"
