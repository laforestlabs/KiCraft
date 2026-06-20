"""Pure leaf-replication remapper: ref/net maps by topology + solved_layout reuse.

Models two structurally-identical stepper axes (rep U1+C1 on STEP_X, sibling
U2+C2 on STEP_Y) sharing +12V/GND rails, and checks that geometry reuse remaps
the per-axis signal while leaving shared rails untouched.
"""
from __future__ import annotations

from kicraft.cli._replicate_leaves import (
    build_replication_maps,
    remap_solved_layout,
)


def _pad(ref, pid, net, x=0.0, y=0.0):
    return {"ref": ref, "pad_id": pid, "net": net,
            "size_mm": {"x": 1.0, "y": 1.0}, "layer": "F.Cu",
            "pos": {"x": x, "y": y}}


def _driver(ref, step_net, pos):
    # 2-pad driver: pad1 = per-axis STEP signal, pad2 = shared GND
    return {
        "ref": ref, "value": "A4988", "width_mm": 5.0, "height_mm": 5.0,
        "is_through_hole": False, "rotation": 90.0,
        "pos": {"x": pos[0], "y": pos[1]},
        "pads": [_pad(ref, "1", step_net), _pad(ref, "2", "GND")],
    }


def _cap(ref, pos):
    # 2-pad cap: pad1 = shared +12V, pad2 = shared GND
    return {
        "ref": ref, "value": "100n", "width_mm": 1.6, "height_mm": 0.8,
        "is_through_hole": False, "rotation": 0.0,
        "pos": {"x": pos[0], "y": pos[1]},
        "pads": [_pad(ref, "1", "+12V"), _pad(ref, "2", "GND")],
    }


def _rep_components():
    return {"U1": _driver("U1", "STEP_X", (10.0, 20.0)), "C1": _cap("C1", (15.0, 22.0))}


def _sib_components():
    # sibling has DIFFERENT refs + per-axis net, SAME footprints/topology, no pos
    return {"U2": _driver("U2", "STEP_Y", (0.0, 0.0)), "C2": _cap("C2", (0.0, 0.0))}


def test_build_maps_happy_path_topology():
    maps = build_replication_maps(
        ["U1", "C1"], ["U2", "C2"], _rep_components(), _sib_components()
    )
    assert maps is not None
    ref_map, net_map = maps
    assert ref_map == {"U1": "U2", "C1": "C2"}
    # per-axis signal remaps; shared rails map to themselves (topology, not name)
    assert net_map == {"STEP_X": "STEP_Y", "GND": "GND", "+12V": "+12V"}


def test_build_maps_rotation_invariant_footprint_match():
    # The solved layout stores POST-rotation geometry: the sibling's driver was
    # placed at 90deg so its width/height are swapped vs the representative's.
    # That is the SAME footprint and must still match (real KC-8AG6FU stepper
    # leaves differ only this way).
    rep = _rep_components()
    sib = _sib_components()
    # swap w/h on the sibling driver to mimic a 90deg-rotated solve
    sib["U2"]["width_mm"], sib["U2"]["height_mm"] = (
        rep["U1"]["height_mm"], rep["U1"]["width_mm"],
    )
    maps = build_replication_maps(["U1", "C1"], ["U2", "C2"], rep, sib)
    assert maps is not None, "rotation-swapped footprint should still match"


def test_build_maps_rejects_component_count_mismatch():
    sib = _sib_components()
    sib["R9"] = _cap("R9", (0.0, 0.0))  # extra part
    assert build_replication_maps(["U1", "C1"], ["U2", "C2", "R9"],
                                  _rep_components(), sib) is None


def test_build_maps_rejects_footprint_mismatch():
    sib = _sib_components()
    sib["U2"]["width_mm"] = 9.9  # different body -> different footprint signature
    assert build_replication_maps(["U1", "C1"], ["U2", "C2"],
                                  _rep_components(), sib) is None


def test_build_maps_rejects_topology_mismatch():
    # sibling wires the cap's pad1 to GND instead of +12V -> no net with the
    # mapped pad-set of rep's +12V -> reject.
    sib = _sib_components()
    sib["C2"]["pads"][0]["net"] = "GND"
    assert build_replication_maps(["U1", "C1"], ["U2", "C2"],
                                  _rep_components(), sib) is None


def test_remap_solved_layout_preserves_geometry_swaps_refs_and_nets():
    rep_layout = {
        "instance_path": "/STEPPER_AXIS_X",
        "sheet_name": "STEPPER_AXIS_X",
        "components": _rep_components(),
        "traces": [
            {"net": "STEP_X", "start": {"x": 1, "y": 2}, "end": {"x": 3, "y": 4}, "width_mm": 0.2},
            {"net": "GND", "start": {"x": 5, "y": 6}, "end": {"x": 7, "y": 8}, "width_mm": 0.2},
        ],
        "vias": [{"net": "+12V", "pos": {"x": 9, "y": 9}, "size_mm": 0.6}],
        "interface_anchors": [{"pad_ref": ["U1", "1"], "port_name": "STEP_X", "pos": {"x": 1, "y": 1}}],
        "ports": [{"name": "STEP_X", "net_name": "STEP_X", "direction": "input"}],
        "bounding_box": {"width_mm": 25.0, "height_mm": 30.0},
    }
    ref_map = {"U1": "U2", "C1": "C2"}
    net_map = {"STEP_X": "STEP_Y", "GND": "GND", "+12V": "+12V"}
    sib_identity = {
        "instance_path": "/STEPPER_AXIS_Y", "sheet_name": "STEPPER_AXIS_Y",
        "sheet_file": "STEPPER_AXIS_Y.kicad_sch",
    }
    out = remap_solved_layout(rep_layout, ref_map, net_map, sib_identity)

    # components rekeyed + refs/pads/nets rewritten, positions preserved
    assert set(out["components"]) == {"U2", "C2"}
    assert out["components"]["U2"]["pos"] == {"x": 10.0, "y": 20.0}  # rep geometry
    assert out["components"]["U2"]["ref"] == "U2"
    assert out["components"]["U2"]["pads"][0]["net"] == "STEP_Y"  # signal remapped
    assert out["components"]["U2"]["pads"][1]["net"] == "GND"     # rail unchanged
    assert out["components"]["U2"]["pads"][0]["ref"] == "U2"
    assert out["components"]["C2"]["pads"][0]["net"] == "+12V"    # rail unchanged
    # traces/vias net-remapped, geometry intact
    assert out["traces"][0]["net"] == "STEP_Y"
    assert out["traces"][0]["start"] == {"x": 1, "y": 2}
    assert out["traces"][1]["net"] == "GND"
    assert out["vias"][0]["net"] == "+12V"
    # anchors + ports
    assert out["interface_anchors"][0]["pad_ref"] == ["U2", "1"]
    assert out["interface_anchors"][0]["port_name"] == "STEP_Y"
    assert out["ports"][0]["name"] == "STEP_Y" and out["ports"][0]["net_name"] == "STEP_Y"
    # identity is the sibling's; geometry summary preserved
    assert out["instance_path"] == "/STEPPER_AXIS_Y"
    assert out["sheet_name"] == "STEPPER_AXIS_Y"
    assert out["bounding_box"] == {"width_mm": 25.0, "height_mm": 30.0}
    assert out["replicated_from"] == "/STEPPER_AXIS_X"
    # the representative layout is untouched (deep copy)
    assert set(rep_layout["components"]) == {"U1", "C1"}
    assert rep_layout["traces"][0]["net"] == "STEP_X"


def test_end_to_end_build_then_remap():
    # derive the maps from topology, then apply -> sibling layout is consistent
    rep_components = _rep_components()
    maps = build_replication_maps(["U1", "C1"], ["U2", "C2"], rep_components, _sib_components())
    assert maps is not None
    ref_map, net_map = maps
    rep_layout = {"instance_path": "/X", "components": rep_components,
                  "traces": [{"net": "STEP_X", "start": {"x": 0, "y": 0}, "end": {"x": 1, "y": 1}}],
                  "vias": [], "interface_anchors": [], "ports": []}
    out = remap_solved_layout(rep_layout, ref_map, net_map, {"instance_path": "/Y"})
    assert out["traces"][0]["net"] == "STEP_Y"
    assert set(out["components"]) == {"U2", "C2"}
