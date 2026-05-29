#!/usr/bin/env python3
"""Stamp a parent (composed) board inside a subprocess.

Invoked by ``compose_subcircuits._stamp_parent_board`` with the
JSON payload path as ``sys.argv[1]``. Lifted out of an inline
string in ``compose_subcircuits.py`` so import-time errors fire
when the file is parsed and so linters / IDEs can see the pcbnew
API calls the way they see normal Python.

Output: the ``__KICRAFT_PCBNEW_OK__`` success sentinel on stdout when the
parent .kicad_pcb is saved.
"""

from __future__ import annotations

import json
import sys

import pcbnew

from kicraft.autoplacer.hardware._pad_nets import propagate_pad_nets


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: _parent_stamp_subprocess.py <payload.json>", file=sys.stderr)
        return 64
    json_path = argv[1]

    with open(json_path) as _f:
        _data = json.load(_f)

    _pcb_path = _data["pcb_path"]
    _out_path = _data["output_path"]
    _components = _data["components"]
    _traces = _data["traces"]
    _vias = _data["vias"]
    _silkscreen = _data.get("silkscreen", []) or []
    _keepouts = _data.get("keepouts", []) or []

    _LAYER_NAME_MAP = {"F.Cu": pcbnew.F_Cu, "B.Cu": pcbnew.B_Cu}
    _SILK_LAYER_MAP = {"F.SilkS": pcbnew.F_SilkS, "B.SilkS": pcbnew.B_SilkS}

    board = pcbnew.LoadBoard(_pcb_path)

    # Snapshot every pcbnew container BEFORE any mutation. KiCad 9's
    # SWIG bindings return non-iterable SwigPyObject wrappers from
    # the second call once the board has been mutated -- captured
    # Python lists are stable.
    _all_drawings = list(board.GetDrawings())
    _all_footprints = list(board.Footprints())
    _all_tracks = list(board.GetTracks())
    _all_zones = list(board.Zones())

    propagate_pad_nets(_all_footprints)

    # Strip every loose drawing -- we rebuild Edge.Cuts + silk +
    # keepouts from the payload below.
    for _d in _all_drawings:
        board.Remove(_d)

    # --- rewrite board outline if provided ---
    _outline = _data.get("outline")
    if _outline:
        _width_mm = max(1.0, _outline["br_x"] - _outline["tl_x"])
        _height_mm = max(1.0, _outline["br_y"] - _outline["tl_y"])
        _left = pcbnew.FromMM(_outline["tl_x"])
        _top = pcbnew.FromMM(_outline["tl_y"])
        _right = pcbnew.FromMM(_outline["tl_x"] + _width_mm)
        _bottom = pcbnew.FromMM(_outline["tl_y"] + _height_mm)

        _corners = [
            (_left, _top),
            (_right, _top),
            (_right, _bottom),
            (_left, _bottom),
        ]
        for _i in range(4):
            _seg = pcbnew.PCB_SHAPE(board)
            _seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
            _seg.SetLayer(pcbnew.Edge_Cuts)
            _seg.SetWidth(pcbnew.FromMM(0.05))
            _x1, _y1 = _corners[_i]
            _x2, _y2 = _corners[(_i + 1) % 4]
            _seg.SetStart(pcbnew.VECTOR2I(_x1, _y1))
            _seg.SetEnd(pcbnew.VECTOR2I(_x2, _y2))
            board.Add(_seg)

    # --- move footprints to composed positions (keep all footprints) ---
    _comp_map = {c["ref"]: c for c in _components}
    for _fp in _all_footprints:
        _ref = _fp.GetReferenceAsString()
        _comp = _comp_map.get(_ref)
        if _comp is None:
            continue
        _cur_layer = 1 if _fp.GetLayer() == pcbnew.B_Cu else 0
        if _comp["layer"] != _cur_layer:
            _fp.Flip(_fp.GetPosition(), False)
        _fp.SetPosition(
            pcbnew.VECTOR2I(pcbnew.FromMM(_comp["x"]), pcbnew.FromMM(_comp["y"]))
        )
        _fp.SetOrientationDegrees(_comp["rotation"])

    # --- clear existing tracks ---
    for _t in _all_tracks:
        board.Remove(_t)

    # --- clear non-rule-area zones ---
    for _z in _all_zones:
        try:
            if _z.GetIsRuleArea():
                continue
        except Exception:
            pass
        board.Remove(_z)

    _netinfo = board.GetNetInfo()

    def _resolve_net(name: str) -> int:
        if not name:
            return 0
        ni = _netinfo.GetNetItem(name)
        if ni is None:
            return 0
        try:
            return int(ni.GetNetCode())
        except Exception:
            return 0

    for _t in _traces:
        _s = pcbnew.PCB_TRACK(board)
        _s.SetStart(
            pcbnew.VECTOR2I(
                pcbnew.FromMM(_t["start_x"]), pcbnew.FromMM(_t["start_y"])
            )
        )
        _s.SetEnd(
            pcbnew.VECTOR2I(
                pcbnew.FromMM(_t["end_x"]), pcbnew.FromMM(_t["end_y"])
            )
        )
        _s.SetLayer(_LAYER_NAME_MAP.get(_t["layer"], pcbnew.F_Cu))
        _s.SetWidth(pcbnew.FromMM(_t["width"]))
        _nc = _resolve_net(_t["net_name"])
        if _nc > 0:
            _s.SetNetCode(_nc)
        board.Add(_s)

    for _v in _vias:
        _tv = pcbnew.PCB_VIA(board)
        _tv.SetPosition(
            pcbnew.VECTOR2I(pcbnew.FromMM(_v["x"]), pcbnew.FromMM(_v["y"]))
        )
        _tv.SetDrill(pcbnew.FromMM(_v["drill"]))
        try:
            _tv.SetWidth(pcbnew.FromMM(_v["size"]))
        except TypeError:
            _tv.SetWidth(pcbnew.F_Cu, pcbnew.FromMM(_v["size"]))
        _nc = _resolve_net(_v["net_name"])
        if _nc > 0:
            _tv.SetNetCode(_nc)
        board.Add(_tv)

    # --- stamp silkscreen graphics ---
    for _silk in _silkscreen:
        _slayer = _SILK_LAYER_MAP.get(_silk.get("layer", "F.SilkS"), pcbnew.F_SilkS)
        if _silk["kind"] == "poly":
            _shape = pcbnew.PCB_SHAPE(board)
            _shape.SetShape(pcbnew.SHAPE_T_POLY)
            _shape.SetLayer(_slayer)
            _shape.SetFilled(False)
            _shape.SetWidth(pcbnew.FromMM(_silk.get("stroke_width", 0.15)))
            _poly = pcbnew.VECTOR_VECTOR2I()
            for _pt in _silk.get("points", []):
                _poly.append(
                    pcbnew.VECTOR2I(pcbnew.FromMM(_pt["x"]), pcbnew.FromMM(_pt["y"]))
                )
            _shape.SetPolyPoints(_poly)
            board.Add(_shape)
        elif _silk["kind"] == "text":
            _txt = pcbnew.PCB_TEXT(board)
            _txt.SetText(_silk.get("text", ""))
            _txt.SetLayer(_slayer)
            _pos = _silk.get("pos", {"x": 0, "y": 0})
            _txt.SetPosition(
                pcbnew.VECTOR2I(pcbnew.FromMM(_pos["x"]), pcbnew.FromMM(_pos["y"]))
            )
            _txt.SetTextSize(
                pcbnew.VECTOR2I(
                    pcbnew.FromMM(_silk.get("font_width", 1.0)),
                    pcbnew.FromMM(_silk.get("font_height", 1.0)),
                )
            )
            _txt.SetTextThickness(pcbnew.FromMM(_silk.get("font_thickness", 0.15)))
            _txt.SetHorizJustify(pcbnew.GR_TEXT_H_ALIGN_LEFT)
            board.Add(_txt)

    # --- stamp parent-local rule-area keepouts (mounting holes etc.) ---
    # These zones survive freerouting_runner.strip_zones() because
    # that helper preserves GetIsRuleArea()==True zones. The DSN
    # export for FreeRouting reads rule-area keepouts so no track or
    # via can be placed inside them.
    _KEEPOUT_LAYERS = [pcbnew.F_Cu, pcbnew.B_Cu]
    for _ko in _keepouts:
        for _layer in _KEEPOUT_LAYERS:
            _zone = pcbnew.ZONE(board)
            _zone.SetLayer(_layer)
            _zone.SetIsRuleArea(True)
            _zone.SetDoNotAllowTracks(True)
            _zone.SetDoNotAllowVias(True)
            # SetDoNotAllowPads(False) on purpose: the keepout rect
            # is bbox(protected_comp) + inward_keep_in_mm, which by
            # construction covers the protected component's own pad.
            # Setting pads not_allowed flagged the protected pad as
            # items_not_allowed against its own zone.
            _zone.SetDoNotAllowPads(False)
            _zone.SetDoNotAllowCopperPour(True)
            _zo = _zone.Outline()
            _zo.NewOutline()
            _x1 = pcbnew.FromMM(_ko["tl_x"])
            _y1 = pcbnew.FromMM(_ko["tl_y"])
            _x2 = pcbnew.FromMM(_ko["br_x"])
            _y2 = pcbnew.FromMM(_ko["br_y"])
            _zo.Append(_x1, _y1)
            _zo.Append(_x2, _y1)
            _zo.Append(_x2, _y2)
            _zo.Append(_x1, _y2)
            board.Add(_zone)

    board.BuildConnectivity()
    board.Save(_out_path)
    # Success sentinel (must match freerouting_runner._PCBNEW_OK_SENTINEL) so a
    # pcbnew/wx teardown SIGSEGV after this successful Save is not mistaken for
    # a stamping failure by _retry_pcbnew_run.
    print("__KICRAFT_PCBNEW_OK__")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
