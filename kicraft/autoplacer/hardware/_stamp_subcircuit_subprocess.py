#!/usr/bin/env python3
"""Stamp a subcircuit board (leaf) inside a subprocess.

Invoked by ``KiCadAdapter.stamp_subcircuit_board_subprocess`` with the
path to a JSON payload as ``sys.argv[1]``. Lives as its own file
(rather than as an inline string in ``adapter.py``) so:

* import-time syntax / name errors surface immediately when adapter.py
  is imported, instead of at runtime when the subprocess crashes;
* linters and IDEs can flag pcbnew API misuse (the previous inline
  version cost us a "AttributeError: 'PCB_TEXT' has no 'GetShape'"
  that silently degraded every leaf solve);
* the smoke test ``tools/smoke_stamp.py`` can exercise this exact
  file with no ``__JSON_PATH__`` substitution dance.

Talks to the parent over the JSON payload only -- everything pcbnew
needs comes from there. Prints ``OK`` on success or ``SELF_LOAD_FAILED``
+ exit code 2 if the saved board fails to round-trip through
``pcbnew.LoadBoard``.
"""

from __future__ import annotations

import hashlib as _hashlib
import json
import os
import shutil as _shutil
import sys
import time as _time
import traceback as _traceback

import pcbnew


def _diag_dump(_path: str) -> dict:
    _info: dict = {"path": _path, "exists": os.path.exists(_path)}
    if not _info["exists"]:
        return _info
    try:
        _info["size"] = os.path.getsize(_path)
        _info["mtime_ns"] = os.stat(_path).st_mtime_ns
        with open(_path, "rb") as _df:
            _data = _df.read()
        _info["sha256"] = _hashlib.sha256(_data).hexdigest()
        _info["nul_bytes"] = _data.count(b"\x00")
        try:
            _text = _data.decode("utf-8", errors="replace")
            _info["paren_balance"] = _text.count("(") - _text.count(")")
            _info["has_nan"] = "nan" in _text.lower()
            _info["has_inf"] = "inf" in _text.lower()
            _lines = _text.splitlines()
            _info["line_count"] = len(_lines)
            _info["head"] = "\n".join(_lines[:5])
            _info["tail"] = "\n".join(_lines[-5:])
        except Exception as _de:
            _info["decode_error"] = repr(_de)
    except OSError as _oe:
        _info["stat_error"] = repr(_oe)
    return _info


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: _stamp_subcircuit_subprocess.py <payload.json>", file=sys.stderr)
        return 64
    json_path = argv[1]

    with open(json_path) as _f:
        _data = json.load(_f)

    _pcb_path = _data["pcb_path"]
    _out_path = _data["output_path"]
    _outline = _data["outline"]
    _components = _data["components"]
    _traces = _data["traces"]
    _vias = _data["vias"]
    _silkscreen = _data.get("silkscreen", []) or []
    _clear_tracks = _data["clear_existing_tracks"]
    _clear_zones = _data["clear_existing_zones"]
    _remove_unmapped = _data["remove_unmapped_footprints"]

    _LAYER_NAME_MAP = {"F.Cu": pcbnew.F_Cu, "B.Cu": pcbnew.B_Cu}
    _SILK_LAYER_MAP = {"F.SilkS": pcbnew.F_SilkS, "B.SilkS": pcbnew.B_SilkS}

    board = pcbnew.LoadBoard(_pcb_path)

    # Snapshot every pcbnew container BEFORE any mutation. KiCad 9's
    # SWIG bindings return non-iterable SwigPyObject wrappers from
    # the second call to GetDrawings()/Footprints()/GetTracks()/
    # Zones() once the board has been mutated, so we capture
    # everything we need to read up front and mutate from those
    # Python lists.
    _all_drawings = list(board.GetDrawings())
    _all_footprints = list(board.Footprints())
    _all_tracks = list(board.GetTracks())
    _all_zones = list(board.Zones())

    # Propagate net codes across same-numbered pads. KiCad treats every pad
    # sharing a number (split thermal pads, dual-terminal tactile switches)
    # as one electrical node, but boards generated through FindPadByNumber
    # leave the duplicate instances on no net; DRC then flags them against
    # the copper that legitimately covers the shared area.
    for _fp in _all_footprints:
        _pads = list(_fp.Pads())
        _net_for_num: dict[str, int] = {}
        for _pad in _pads:
            _nc = _pad.GetNetCode()
            if _nc:
                _net_for_num.setdefault(_pad.GetNumber(), _nc)
        for _pad in _pads:
            if _pad.GetNetCode() == 0:
                _nc = _net_for_num.get(_pad.GetNumber())
                if _nc:
                    _pad.SetNetCode(_nc)

    # --- rewrite board outline (Edge.Cuts) ---
    # Strip every loose drawing -- we rebuild Edge.Cuts + silk from
    # the payload below.
    for d in _all_drawings:
        board.Remove(d)

    _polyline_mm = _outline.get("polyline")
    if _polyline_mm is not None and len(_polyline_mm) >= 3:
        # Leaf flow: Edge.Cuts traces the SAME closed polyline as the
        # F.SilkS leaf outline. The polyline came from
        # subcircuit_solver.leaf_outline_polyline upstream; we just
        # stamp segments between consecutive points (and a closing
        # segment from last to first) so the two layers share one
        # contour and cannot drift.
        _n = len(_polyline_mm)
        for _i in range(_n):
            _x1_mm, _y1_mm = _polyline_mm[_i]
            _x2_mm, _y2_mm = _polyline_mm[(_i + 1) % _n]
            _edge = pcbnew.PCB_SHAPE(board)
            _edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
            _edge.SetLayer(pcbnew.Edge_Cuts)
            _edge.SetWidth(pcbnew.FromMM(0.05))
            _edge.SetStart(
                pcbnew.VECTOR2I(pcbnew.FromMM(_x1_mm), pcbnew.FromMM(_y1_mm))
            )
            _edge.SetEnd(
                pcbnew.VECTOR2I(pcbnew.FromMM(_x2_mm), pcbnew.FromMM(_y2_mm))
            )
            board.Add(_edge)
    else:
        # Parent flow / unlabeled leaf: sharp 4-segment rectangle.
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
            _x1, _y1 = _corners[_i]
            _x2, _y2 = _corners[(_i + 1) % 4]
            _edge = pcbnew.PCB_SHAPE(board)
            _edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
            _edge.SetLayer(pcbnew.Edge_Cuts)
            _edge.SetWidth(pcbnew.FromMM(0.05))
            _edge.SetStart(pcbnew.VECTOR2I(_x1, _y1))
            _edge.SetEnd(pcbnew.VECTOR2I(_x2, _y2))
            board.Add(_edge)

    # --- move / remove footprints to match the placement payload ---
    _comp_map = {c["ref"]: c for c in _components}
    for _fp in _all_footprints:
        _ref = _fp.GetReferenceAsString()
        _comp = _comp_map.get(_ref)
        if _comp is None:
            if _remove_unmapped:
                board.Remove(_fp)
            continue
        if _fp.IsLocked():
            continue
        _cur_layer = 1 if _fp.GetLayer() == pcbnew.B_Cu else 0
        if _comp["layer"] != _cur_layer:
            _fp.Flip(_fp.GetPosition(), False)
        _fp.SetPosition(
            pcbnew.VECTOR2I(pcbnew.FromMM(_comp["x"]), pcbnew.FromMM(_comp["y"]))
        )
        _fp.SetOrientationDegrees(_comp["rotation"])

    if _clear_tracks:
        for _t in _all_tracks:
            board.Remove(_t)

    if _clear_zones:
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

    # Silk for this leaf arrives in the payload's "silkscreen" list
    # (built by leaf_routing._silk_for_leaf against the post-repair
    # component bbox). Drawing it here -- or deriving any outline silk
    # from the Edge.Cuts rectangle -- would either duplicate that
    # rounded poly or compete with it as a sharp-corner rectangle (a
    # previous regression). The leaf solver owns the silk shape; this
    # code only owns Edge.Cuts.
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

    # Do NOT call board.BuildConnectivity() here: pcbnew Save() silently
    # returns False (no file written, no exception) on ~half of attempts
    # when called after BuildConnectivity on a heavily-mutated board.
    # Consumers rebuild connectivity on LoadBoard().

    _save_status = board.Save(_out_path)

    # pcbnew.Save() emits a sidecar .kicad_pro with KiCad defaults.
    # Overwrite with the source PCB's pro so kicad-cli pcb drc sees
    # the project's actual netclass / rules.
    _src_pro = os.path.splitext(_pcb_path)[0] + ".kicad_pro"
    _dst_pro = os.path.splitext(_out_path)[0] + ".kicad_pro"
    if (
        os.path.exists(_src_pro)
        and os.path.abspath(_src_pro) != os.path.abspath(_dst_pro)
    ):
        try:
            _shutil.copy2(_src_pro, _dst_pro)
        except OSError:
            pass

    try:
        with open(_out_path, "rb") as _tf:
            os.fsync(_tf.fileno())
    except OSError:
        pass
    try:
        _out_dir = os.path.dirname(_out_path) or "."
        _dir_fd = os.open(_out_dir, os.O_DIRECTORY)
        try:
            os.fsync(_dir_fd)
        finally:
            os.close(_dir_fd)
    except OSError:
        pass

    # Self-load validator. pcbnew.LoadBoard() returns None instead of
    # raising on parser rejection, so catch that here and capture the
    # bad artifact before the next round overwrites it.
    _self_load_ok = False
    _self_load_err = None
    try:
        _verify = pcbnew.LoadBoard(_out_path)
        _self_load_ok = _verify is not None
    except Exception as _se:
        _self_load_err = "".join(
            _traceback.format_exception_only(type(_se), _se)
        ).strip()

    if not _self_load_ok:
        _capture_root = os.path.join(
            os.path.dirname(_out_path) or ".", ".failed_capture"
        )
        _stamp = _time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        _capture_dir = os.path.join(_capture_root, _stamp)
        try:
            os.makedirs(_capture_dir, exist_ok=True)
            _base = os.path.splitext(_out_path)[0]
            _diag: dict = {
                "save_status": repr(_save_status),
                "self_load_ok": _self_load_ok,
                "self_load_error": _self_load_err,
                "out_path": _out_path,
                "pcb_path": _pcb_path,
                "captured_at": _stamp,
                "files": {},
            }
            for _ext in (".kicad_pcb", ".kicad_pro", ".kicad_prl"):
                _src = _base + _ext
                _diag["files"][_ext] = _diag_dump(_src)
                if os.path.exists(_src):
                    try:
                        _shutil.copy2(_src, os.path.join(_capture_dir, "leaf" + _ext))
                    except OSError as _ce:
                        _diag["files"][_ext]["copy_error"] = repr(_ce)
            try:
                _shutil.copy2(json_path, os.path.join(_capture_dir, "stamp_payload.json"))
            except OSError as _je:
                _diag["payload_copy_error"] = repr(_je)
            with open(os.path.join(_capture_dir, "diagnostics.json"), "w") as _df:
                json.dump(_diag, _df, indent=2, default=str)
        except OSError as _ce:
            print(f"DIAG_CAPTURE_FAILED {_ce!r}")
        print(
            f"SELF_LOAD_FAILED save_status={_save_status!r} "
            f"capture_dir={_capture_dir} self_load_error={_self_load_err!r}"
        )
        return 2

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
