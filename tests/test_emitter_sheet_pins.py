"""WS11: root sheet pins must each get a distinct on-grid Y.

The old ``height/(n_pins+1)`` step fell below the 1.27 mm grid for pin-dense
sheets (24 pins on a 30.48 mm sheet -> 1.22 mm), so adjacent pins snapped to the
SAME Y and their nets merged silently -- caught only by the netlist-faithfulness
diff.
"""

from __future__ import annotations

from kicraft.design.models import InterSheetNet, Sheet, SheetPin
from kicraft.design.synthesis.emitter import (
    _SHEET_PIN_PITCH_MM,
    _SheetInstance,
    _emit_sheet_block,
    _sheet_required_height,
)


def _sheet_inst(n_signal_pins: int, name: str = "MCU") -> _SheetInstance:
    endpoints = []
    for i in range(n_signal_pins):
        net = InterSheetNet(
            name=f"SIGNAL_{i}",
            endpoints=[
                SheetPin(sheet=name, direction="bidirectional"),
                SheetPin(sheet="OTHER", direction="bidirectional"),
            ],
        )
        endpoints.append((net, net.endpoints[0]))
    sheet = Sheet(name=name, stem=name.replace(" ", "_"), function="test")
    return _SheetInstance(
        sheet=sheet, instance_uuid="u1", leaf_uuid="u2", parts=[],
        inter_sheet_endpoints=endpoints,
    )


def test_24_signal_pins_get_distinct_y():
    si = _sheet_inst(24)
    height = _sheet_required_height(si, base_height=30.48)
    _block, recs = _emit_sheet_block(
        si, x=0.0, y=0.0, width=38.1, height=height, project_stem="proj"
    )
    ys = [py for (_net, _px, py) in recs]
    assert len(ys) == 24
    assert len(set(ys)) == 24  # all distinct -- no silent net merge

    ys_sorted = sorted(ys)
    for a, b in zip(ys_sorted, ys_sorted[1:]):
        assert abs((b - a) - _SHEET_PIN_PITCH_MM) < 1e-9  # exact 1.27 mm pitch


def test_height_grows_for_pin_dense_sheet_only():
    # 24 pins need 25 slots * 1.27 = 31.75 mm > the 30.48 mm base -> grow.
    dense = _sheet_inst(24)
    assert _sheet_required_height(dense, 30.48) == _SHEET_PIN_PITCH_MM * 25
    # A sparse sheet keeps the base height (byte-for-byte unchanged layout).
    sparse = _sheet_inst(3)
    assert _sheet_required_height(sparse, 30.48) == 30.48
