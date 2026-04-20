"""Tests for kicraft.autoplacer.brain.subcircuit_instances -- normalize-early parsers."""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.types import (
    Component,
    InterfaceAnchor,
    Layer,
    Point,
    TraceSegment,
    Via,
)
from kicraft.autoplacer.brain.subcircuit_instances import (
    _normalize_to_canonical,
    _parse_components,
    _parse_traces,
    _parse_vias,
    _parse_interface_anchors,
    _parse_bbox,
    _parse_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_component_dict(
    ref="R1", value="10k", x=1.0, y=2.0, rotation=0.0,
    layer="F.Cu", width_mm=1.6, height_mm=0.8, pads=None,
):
    """Return a serialized component dict matching artifact format."""
    d = {
        "ref": ref,
        "value": value,
        "pos": {"x": x, "y": y},
        "rotation": rotation,
        "layer": layer,
        "width_mm": width_mm,
        "height_mm": height_mm,
    }
    if pads is not None:
        d["pads"] = pads
    return d


def _make_trace_dict(
    sx=0.0, sy=0.0, ex=3.0, ey=4.0,
    layer="F.Cu", net="CLK", width_mm=0.2,
):
    return {
        "start": {"x": sx, "y": sy},
        "end": {"x": ex, "y": ey},
        "layer": layer,
        "net": net,
        "width_mm": width_mm,
    }


def _make_via_dict(x=5.0, y=6.0, net="VCC", drill_mm=0.3, size_mm=0.6):
    return {
        "pos": {"x": x, "y": y},
        "net": net,
        "drill_mm": drill_mm,
        "size_mm": size_mm,
    }


def _make_anchor_dict(port_name="A", x=1.0, y=2.0, layer="F.Cu", pad_ref=None):
    d = {
        "port_name": port_name,
        "pos": {"x": x, "y": y},
        "layer": layer,
    }
    if pad_ref is not None:
        d["pad_ref"] = pad_ref
    return d


# ---- _normalize_to_canonical -----------------------------------------------

class TestNormalizeToCanonical:
    """Tests for _normalize_to_canonical()."""

    def test_solved_layout_returned_as_is(self):
        """When solved_layout is a non-empty dict, return it directly."""
        solved = {
            "components": {"R1": _make_component_dict()},
            "traces": [],
            "vias": [],
            "ports": [],
            "interface_anchors": [],
            "bounding_box": {},
            "score": 85.0,
        }
        result = _normalize_to_canonical({}, {}, solved_layout=solved)
        assert result is solved

    def test_solved_layout_empty_dict_falls_through(self):
        """An empty solved_layout dict triggers fallback path."""
        result = _normalize_to_canonical({}, {}, solved_layout={})
        assert "components" in result
        assert result["components"] == {}

    def test_solved_layout_none_falls_through(self):
        result = _normalize_to_canonical({}, {}, solved_layout=None)
        assert result["components"] == {}

    def test_components_from_solved_components(self):
        """Components first try debug[solved_components]."""
        debug = {"solved_components": {"R1": _make_component_dict()}}
        result = _normalize_to_canonical({}, debug)
        assert "R1" in result["components"]

    def test_components_invalid_type_gives_empty(self):
        """Non-dict solved_components produces empty dict."""
        debug = {"solved_components": "not a dict"}
        result = _normalize_to_canonical({}, debug)
        assert result["components"] == {}

    def test_traces_empty_without_solved_layout(self):
        """Without solved_layout, traces and vias are empty."""
        debug = {"extra": {}}
        result = _normalize_to_canonical({}, debug)
        assert result["traces"] == []
        assert result["vias"] == []

    def test_ports_from_metadata(self):
        metadata = {"interface_ports": [{"name": "SDA", "net_name": "SDA_NET"}]}
        result = _normalize_to_canonical(metadata, {})
        assert len(result["ports"]) == 1

    def test_interface_anchors_level1_solve_summary_placement_result(self):
        """First priority: debug.extra.solve_summary.placement_result.interface_anchors."""
        anchors = [_make_anchor_dict(port_name="P1")]
        debug = {
            "extra": {
                "solve_summary": {
                    "placement_result": {"interface_anchors": anchors}
                }
            }
        }
        result = _normalize_to_canonical({}, debug)
        assert len(result["interface_anchors"]) == 1
        assert result["interface_anchors"][0]["port_name"] == "P1"

    def test_interface_anchors_level2_placement_result(self):
        """Second priority: debug.extra.placement_result.interface_anchors."""
        anchors = [_make_anchor_dict(port_name="P2")]
        debug = {
            "extra": {
                "placement_result": {"interface_anchors": anchors}
            }
        }
        result = _normalize_to_canonical({}, debug)
        assert len(result["interface_anchors"]) == 1
        assert result["interface_anchors"][0]["port_name"] == "P2"

    def test_interface_anchors_level3_solve_summary_direct(self):
        """Third priority: debug.extra.solve_summary.interface_anchors."""
        anchors = [_make_anchor_dict(port_name="P3")]
        debug = {
            "extra": {
                "solve_summary": {"interface_anchors": anchors}
            }
        }
        result = _normalize_to_canonical({}, debug)
        assert len(result["interface_anchors"]) == 1
        assert result["interface_anchors"][0]["port_name"] == "P3"

    def test_flat_xy_anchors_normalized_to_pos(self):
        """Anchors with flat x/y keys get normalized to nested pos dict."""
        debug = {
            "extra": {
                "solve_summary": {
                    "interface_anchors": [
                        {"x": 1.0, "y": 2.0, "port_name": "A", "layer": "F.Cu"},
                    ]
                }
            }
        }
        result = _normalize_to_canonical({}, debug)
        anchor = result["interface_anchors"][0]
        assert "pos" in anchor
        assert anchor["pos"] == {"x": 1.0, "y": 2.0}
        assert "x" not in anchor
        assert "y" not in anchor

    def test_anchors_non_dict_entries_skipped(self):
        debug = {
            "extra": {
                "solve_summary": {
                    "interface_anchors": [
                        _make_anchor_dict(port_name="OK"),
                        "not a dict",
                        42,
                        None,
                    ]
                }
            }
        }
        result = _normalize_to_canonical({}, debug)
        assert len(result["interface_anchors"]) == 1

    def test_bounding_box_from_metadata(self):
        metadata = {
            "local_board_outline": {"width_mm": 20.0, "height_mm": 15.0}
        }
        result = _normalize_to_canonical(metadata, {})
        assert result["bounding_box"]["width_mm"] == pytest.approx(20.0)
        assert result["bounding_box"]["height_mm"] == pytest.approx(15.0)

    def test_bounding_box_fallback_to_debug_leaf_extraction(self):
        debug = {
            "leaf_extraction": {
                "local_board_outline": {"width_mm": 10.0, "height_mm": 8.0}
            }
        }
        result = _normalize_to_canonical({}, debug)
        assert result["bounding_box"]["width_mm"] == pytest.approx(10.0)
        assert result["bounding_box"]["height_mm"] == pytest.approx(8.0)

    def test_bounding_box_missing_gives_empty_dict(self):
        result = _normalize_to_canonical({}, {})
        assert result["bounding_box"] == {}

    def test_score_from_best_round(self):
        debug = {"extra": {"best_round": {"score": 92.5}}}
        result = _normalize_to_canonical({}, debug)
        assert result["score"] == pytest.approx(92.5)

    def test_score_fallback_to_solve_summary_best_round(self):
        debug = {
            "extra": {
                "solve_summary": {"best_round": {"score": 78.0}}
            }
        }
        result = _normalize_to_canonical({}, debug)
        assert result["score"] == pytest.approx(78.0)

    def test_score_missing_gives_none(self):
        result = _normalize_to_canonical({}, {})
        assert result["score"] is None

    def test_all_canonical_keys_present(self):
        """Verify every expected key is present in result."""
        result = _normalize_to_canonical({}, {})
        for key in ("components", "traces", "vias", "ports",
                    "interface_anchors", "bounding_box", "score"):
            assert key in result


# ---- _parse_components ------------------------------------------------------

class TestParseComponents:
    """Tests for _parse_components()."""

    def test_valid_components(self):
        canonical = {
            "components": {
                "R1": _make_component_dict(ref="R1", value="10k", x=5.0, y=10.0),
                "C1": _make_component_dict(ref="C1", value="100nF", x=15.0, y=20.0),
            }
        }
        comps = _parse_components(canonical)
        assert len(comps) == 2
        assert "R1" in comps and "C1" in comps
        assert isinstance(comps["R1"], Component)
        assert comps["R1"].ref == "R1"
        assert comps["R1"].value == "10k"
        assert comps["R1"].pos.x == pytest.approx(5.0)
        assert comps["R1"].pos.y == pytest.approx(10.0)

    def test_component_layer_back(self):
        canonical = {
            "components": {
                "U1": _make_component_dict(ref="U1", layer="B.Cu"),
            }
        }
        comps = _parse_components(canonical)
        assert comps["U1"].layer == Layer.BACK

    def test_component_with_pads(self):
        pads = [
            {"ref": "R1", "pad_id": "1", "pos": {"x": 0.0, "y": 0.0}, "net": "A", "layer": "F.Cu"},
            {"ref": "R1", "pad_id": "2", "pos": {"x": 1.6, "y": 0.0}, "net": "B", "layer": "F.Cu"},
        ]
        canonical = {
            "components": {
                "R1": _make_component_dict(ref="R1", pads=pads),
            }
        }
        comps = _parse_components(canonical)
        assert len(comps["R1"].pads) == 2
        assert comps["R1"].pads[0].pad_id == "1"
        assert comps["R1"].pads[1].net == "B"

    def test_empty_components(self):
        assert _parse_components({"components": {}}) == {}

    def test_missing_components_key(self):
        assert _parse_components({}) == {}

    def test_non_dict_components_value(self):
        assert _parse_components({"components": "bad"}) == {}
        assert _parse_components({"components": [1, 2]}) == {}

    def test_non_dict_entries_skipped(self):
        canonical = {
            "components": {
                "R1": _make_component_dict(ref="R1"),
                "BAD": "not a dict",
                "ALSO_BAD": 42,
            }
        }
        comps = _parse_components(canonical)
        assert len(comps) == 1
        assert "R1" in comps

    def test_component_optional_fields(self):
        comp_dict = _make_component_dict(ref="J1")
        comp_dict["kind"] = "connector"
        comp_dict["is_through_hole"] = True
        comp_dict["locked"] = True
        comp_dict["body_center"] = {"x": 1.5, "y": 2.5}
        comp_dict["opening_direction"] = 90.0
        canonical = {"components": {"J1": comp_dict}}
        comps = _parse_components(canonical)
        j1 = comps["J1"]
        assert j1.kind == "connector"
        assert j1.is_through_hole is True
        assert j1.locked is True
        assert j1.body_center.x == pytest.approx(1.5)
        assert j1.body_center.y == pytest.approx(2.5)
        assert j1.opening_direction == pytest.approx(90.0)


# ---- _parse_traces ----------------------------------------------------------

class TestParseTraces:
    """Tests for _parse_traces()."""

    def test_valid_traces(self):
        canonical = {
            "traces": [
                _make_trace_dict(sx=0, sy=0, ex=3, ey=4, layer="F.Cu", net="CLK", width_mm=0.2),
                _make_trace_dict(sx=1, sy=1, ex=5, ey=5, layer="B.Cu", net="GND", width_mm=0.25),
            ]
        }
        traces = _parse_traces(canonical)
        assert len(traces) == 2
        assert isinstance(traces[0], TraceSegment)
        assert traces[0].start.x == pytest.approx(0.0)
        assert traces[0].end.y == pytest.approx(4.0)
        assert traces[0].layer == Layer.FRONT
        assert traces[0].net == "CLK"
        assert traces[0].width_mm == pytest.approx(0.2)
        assert traces[1].layer == Layer.BACK

    def test_default_width(self):
        canonical = {
            "traces": [
                {
                    "start": {"x": 0, "y": 0},
                    "end": {"x": 1, "y": 1},
                    "layer": "F.Cu",
                    "net": "SIG",
                }
            ]
        }
        traces = _parse_traces(canonical)
        assert len(traces) == 1
        assert traces[0].width_mm == pytest.approx(0.127)

    def test_non_dict_entries_skipped(self):
        canonical = {
            "traces": [
                _make_trace_dict(),
                "not a dict",
                42,
                None,
            ]
        }
        traces = _parse_traces(canonical)
        assert len(traces) == 1

    def test_empty_list(self):
        assert _parse_traces({"traces": []}) == []

    def test_missing_traces_key(self):
        assert _parse_traces({}) == []

    def test_non_list_traces(self):
        assert _parse_traces({"traces": "bad"}) == []
        assert _parse_traces({"traces": 42}) == []


# ---- _parse_vias ------------------------------------------------------------

class TestParseVias:
    """Tests for _parse_vias()."""

    def test_valid_vias(self):
        canonical = {
            "vias": [
                _make_via_dict(x=5, y=6, net="VCC", drill_mm=0.4, size_mm=0.8),
            ]
        }
        vias = _parse_vias(canonical)
        assert len(vias) == 1
        assert isinstance(vias[0], Via)
        assert vias[0].pos.x == pytest.approx(5.0)
        assert vias[0].pos.y == pytest.approx(6.0)
        assert vias[0].net == "VCC"
        assert vias[0].drill_mm == pytest.approx(0.4)
        assert vias[0].size_mm == pytest.approx(0.8)

    def test_default_sizes(self):
        canonical = {
            "vias": [
                {"pos": {"x": 1, "y": 2}, "net": "GND"}
            ]
        }
        vias = _parse_vias(canonical)
        assert len(vias) == 1
        assert vias[0].drill_mm == pytest.approx(0.3)
        assert vias[0].size_mm == pytest.approx(0.6)

    def test_non_dict_entries_skipped(self):
        canonical = {
            "vias": [
                _make_via_dict(),
                "bad",
                123,
                None,
            ]
        }
        vias = _parse_vias(canonical)
        assert len(vias) == 1

    def test_empty_list(self):
        assert _parse_vias({"vias": []}) == []

    def test_missing_vias_key(self):
        assert _parse_vias({}) == []

    def test_non_list_vias(self):
        assert _parse_vias({"vias": "bad"}) == []


# ---- _parse_interface_anchors -----------------------------------------------

class TestParseInterfaceAnchors:
    """Tests for _parse_interface_anchors()."""

    def test_valid_anchors(self):
        canonical = {
            "interface_anchors": [
                _make_anchor_dict(port_name="SDA", x=10.0, y=20.0, layer="F.Cu"),
            ]
        }
        anchors = _parse_interface_anchors(canonical)
        assert len(anchors) == 1
        assert isinstance(anchors[0], InterfaceAnchor)
        assert anchors[0].port_name == "SDA"
        assert anchors[0].pos.x == pytest.approx(10.0)
        assert anchors[0].pos.y == pytest.approx(20.0)
        assert anchors[0].layer == Layer.FRONT

    def test_anchor_back_layer(self):
        canonical = {
            "interface_anchors": [
                _make_anchor_dict(port_name="P1", layer="B.Cu"),
            ]
        }
        anchors = _parse_interface_anchors(canonical)
        assert anchors[0].layer == Layer.BACK

    def test_pad_ref_valid_list(self):
        canonical = {
            "interface_anchors": [
                _make_anchor_dict(port_name="P1", pad_ref=["U1", "3"]),
            ]
        }
        anchors = _parse_interface_anchors(canonical)
        assert anchors[0].pad_ref == ("U1", "3")

    def test_pad_ref_wrong_length_gives_none(self):
        canonical = {
            "interface_anchors": [
                _make_anchor_dict(port_name="P1", pad_ref=["U1"]),
            ]
        }
        anchors = _parse_interface_anchors(canonical)
        assert anchors[0].pad_ref is None

    def test_pad_ref_three_elements_gives_none(self):
        canonical = {
            "interface_anchors": [
                _make_anchor_dict(port_name="P1", pad_ref=["U1", "3", "extra"]),
            ]
        }
        anchors = _parse_interface_anchors(canonical)
        assert anchors[0].pad_ref is None

    def test_pad_ref_non_list_gives_none(self):
        canonical = {
            "interface_anchors": [
                _make_anchor_dict(port_name="P1", pad_ref="not_a_list"),
            ]
        }
        anchors = _parse_interface_anchors(canonical)
        assert anchors[0].pad_ref is None

    def test_non_dict_entries_skipped(self):
        canonical = {
            "interface_anchors": [
                _make_anchor_dict(port_name="GOOD"),
                "bad",
                42,
                None,
            ]
        }
        anchors = _parse_interface_anchors(canonical)
        assert len(anchors) == 1

    def test_empty_list(self):
        assert _parse_interface_anchors({"interface_anchors": []}) == []

    def test_missing_key(self):
        assert _parse_interface_anchors({}) == []

    def test_non_list_value(self):
        assert _parse_interface_anchors({"interface_anchors": "bad"}) == []


# ---- _parse_bbox ------------------------------------------------------------

class TestParseBbox:
    """Tests for _parse_bbox()."""

    def test_explicit_bbox(self):
        canonical = {
            "bounding_box": {"width_mm": 20.0, "height_mm": 15.0},
        }
        w, h = _parse_bbox(canonical, {})
        assert w == pytest.approx(20.0)
        assert h == pytest.approx(15.0)

    def test_bbox_from_components_fallback(self):
        """When no explicit bbox, compute from component positions."""
        comps = {
            "R1": Component(
                ref="R1", value="10k",
                pos=Point(10.0, 10.0), rotation=0.0,
                layer=Layer.FRONT, width_mm=2.0, height_mm=1.0,
            ),
            "R2": Component(
                ref="R2", value="10k",
                pos=Point(20.0, 20.0), rotation=0.0,
                layer=Layer.FRONT, width_mm=2.0, height_mm=1.0,
            ),
        }
        w, h = _parse_bbox({}, comps)
        # R1 bbox: (9,9.5)-(11,10.5), R2 bbox: (19,19.5)-(21,20.5)
        assert w == pytest.approx(12.0)  # 21 - 9
        assert h == pytest.approx(11.0)  # 20.5 - 9.5

    def test_no_bbox_no_components(self):
        w, h = _parse_bbox({}, {})
        assert w == pytest.approx(0.0)
        assert h == pytest.approx(0.0)

    def test_empty_bounding_box_dict_falls_through(self):
        """Empty bounding_box dict has no width_mm, so falls back."""
        w, h = _parse_bbox({"bounding_box": {}}, {})
        assert w == pytest.approx(0.0)
        assert h == pytest.approx(0.0)

    def test_non_dict_bounding_box_falls_through(self):
        w, h = _parse_bbox({"bounding_box": "bad"}, {})
        assert w == pytest.approx(0.0)
        assert h == pytest.approx(0.0)


# ---- _parse_score -----------------------------------------------------------

class TestParseScore:
    """Tests for _parse_score()."""

    def test_int(self):
        assert _parse_score(85) == pytest.approx(85.0)

    def test_float(self):
        assert _parse_score(92.5) == pytest.approx(92.5)

    def test_string_numeric(self):
        assert _parse_score("78.3") == pytest.approx(78.3)

    def test_none(self):
        assert _parse_score(None) == pytest.approx(0.0)

    def test_non_convertible_string(self):
        assert _parse_score("not a number") == pytest.approx(0.0)

    def test_empty_string(self):
        assert _parse_score("") == pytest.approx(0.0)

    def test_zero(self):
        assert _parse_score(0) == pytest.approx(0.0)

    def test_negative(self):
        assert _parse_score(-10.5) == pytest.approx(-10.5)
