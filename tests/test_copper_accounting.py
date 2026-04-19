"""Tests for kicraft.autoplacer.brain.copper_accounting -- copper preservation tracking.

All tests use lightweight mock objects; no pcbnew dependency.
"""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.copper_accounting import (
    CopperManifest,
    build_copper_manifest,
    fingerprint_trace,
    fingerprint_via,
    verify_copper_preservation,
)


# ---------------------------------------------------------------------------
# Lightweight mock types
# ---------------------------------------------------------------------------


class MockPoint:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class MockLayer:
    __slots__ = ("name",)

    def __init__(self, name: str = "F.Cu"):
        self.name = name


class MockTraceSegment:
    __slots__ = ("start", "end", "layer", "net", "width_mm")

    def __init__(
        self,
        start: MockPoint,
        end: MockPoint,
        layer: MockLayer | None = None,
        net: str = "NET",
        width_mm: float = 0.25,
    ):
        self.start = start
        self.end = end
        self.layer = layer or MockLayer()
        self.net = net
        self.width_mm = width_mm


class MockVia:
    __slots__ = ("pos", "net", "drill_mm", "size_mm")

    def __init__(
        self,
        pos: MockPoint,
        net: str = "NET",
        drill_mm: float = 0.3,
        size_mm: float = 0.6,
    ):
        self.pos = pos
        self.net = net
        self.drill_mm = drill_mm
        self.size_mm = size_mm


class MockLayoutId:
    __slots__ = ("sheet_name",)

    def __init__(self, sheet_name: str):
        self.sheet_name = sheet_name


class MockInstance:
    __slots__ = ("layout_id",)

    def __init__(self, sheet_name: str):
        self.layout_id = MockLayoutId(sheet_name)


class MockTransformed:
    __slots__ = ("transformed_traces", "transformed_vias")

    def __init__(
        self,
        traces: list | None = None,
        vias: list | None = None,
    ):
        self.transformed_traces = traces or []
        self.transformed_vias = vias or []


class MockComposedChild:
    __slots__ = ("_instance_path", "instance", "transformed")

    def __init__(
        self,
        instance_path: str,
        sheet_name: str,
        traces: list | None = None,
        vias: list | None = None,
    ):
        self._instance_path = instance_path
        self.instance = MockInstance(sheet_name)
        self.transformed = MockTransformed(traces, vias)

    @property
    def instance_path(self) -> str:
        return self._instance_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trace(
    x1: float, y1: float, x2: float, y2: float,
    layer: str = "F.Cu", net: str = "NET", width: float = 0.25,
) -> MockTraceSegment:
    return MockTraceSegment(
        start=MockPoint(x1, y1),
        end=MockPoint(x2, y2),
        layer=MockLayer(layer),
        net=net,
        width_mm=width,
    )


def _via(
    x: float, y: float,
    net: str = "NET", drill: float = 0.3, size: float = 0.6,
) -> MockVia:
    return MockVia(MockPoint(x, y), net=net, drill_mm=drill, size_mm=size)


# ---------------------------------------------------------------------------
# fingerprint_trace
# ---------------------------------------------------------------------------


def test_fingerprint_trace_with_object():
    t = _trace(1.0, 2.0, 3.0, 4.0, layer="F.Cu", width=0.25)
    fp = fingerprint_trace(t)
    assert fp == (1.0, 2.0, 3.0, 4.0, "F.Cu", 0.25)


def test_fingerprint_trace_rounds_coordinates():
    """FP jitter within 0.01 mm rounds to the same fingerprint."""
    t = _trace(1.001, 2.004, 3.009, 4.005, width=0.2501)
    fp = fingerprint_trace(t)
    assert fp == (1.0, 2.0, 3.01, 4.0, "F.Cu", 0.25)


def test_fingerprint_trace_with_dict():
    d = {
        "start_x": 1.0,
        "start_y": 2.0,
        "end_x": 3.0,
        "end_y": 4.0,
        "layer": "B.Cu",
        "width_mm": 0.15,
    }
    fp = fingerprint_trace(d)
    assert fp == (1.0, 2.0, 3.0, 4.0, "B.Cu", 0.15)


def test_fingerprint_trace_dict_uses_width_key():
    """Dict path falls back to "width" when "width_mm" is absent."""
    d = {"start_x": 0, "start_y": 0, "end_x": 5, "end_y": 0, "layer": "F.Cu", "width": 0.2}
    fp = fingerprint_trace(d)
    assert fp == (0.0, 0.0, 5.0, 0.0, "F.Cu", 0.2)


# ---------------------------------------------------------------------------
# fingerprint_via
# ---------------------------------------------------------------------------


def test_fingerprint_via_with_object():
    v = _via(10.0, 20.0, drill=0.4, size=0.8)
    fp = fingerprint_via(v)
    assert fp == (10.0, 20.0, 0.4, 0.8)


def test_fingerprint_via_rounds_coordinates():
    v = _via(10.004, 20.006, drill=0.3001, size=0.5999)
    fp = fingerprint_via(v)
    assert fp == (10.0, 20.01, 0.3, 0.6)


def test_fingerprint_via_with_dict():
    d = {"x": 5.0, "y": 6.0, "drill_mm": 0.3, "size_mm": 0.6}
    fp = fingerprint_via(d)
    assert fp == (5.0, 6.0, 0.3, 0.6)


def test_fingerprint_via_dict_alt_keys():
    """Dict path accepts "drill" and "size" as key aliases."""
    d = {"x": 1.0, "y": 2.0, "drill": 0.35, "size": 0.7}
    fp = fingerprint_via(d)
    assert fp == (1.0, 2.0, 0.35, 0.7)


# ---------------------------------------------------------------------------
# build_copper_manifest
# ---------------------------------------------------------------------------


def test_build_manifest_single_child():
    t1 = _trace(0, 0, 3, 4, net="A")
    v1 = _via(1.5, 2.0, net="A")
    child = MockComposedChild("/child_a", "CHILD_A", traces=[t1], vias=[v1])

    manifest = build_copper_manifest([child])

    assert manifest.total_child_traces == 1
    assert manifest.total_child_vias == 1
    assert manifest.total_child_length_mm == pytest.approx(5.0)
    assert "/child_a" in manifest.per_child
    entry = manifest.per_child["/child_a"]
    assert entry.sheet_name == "CHILD_A"
    assert entry.trace_count == 1
    assert entry.via_count == 1
    assert len(entry.trace_fingerprints) == 1
    assert len(entry.via_fingerprints) == 1


def test_build_manifest_multiple_children():
    t1 = _trace(0, 0, 10, 0, net="SIG")
    t2 = _trace(0, 0, 0, 5, net="PWR")
    child_a = MockComposedChild("/a", "CHILD_A", traces=[t1], vias=[])
    child_b = MockComposedChild("/b", "CHILD_B", traces=[t2], vias=[_via(0, 2.5)])

    manifest = build_copper_manifest([child_a, child_b])

    assert manifest.total_child_traces == 2
    assert manifest.total_child_vias == 1
    assert len(manifest.per_child) == 2
    assert manifest.per_child["/a"].trace_count == 1
    assert manifest.per_child["/b"].trace_count == 1
    assert manifest.per_child["/b"].via_count == 1


def test_build_manifest_with_parent_traces():
    child = MockComposedChild("/c", "CHILD", traces=[_trace(0, 0, 1, 0)], vias=[])
    parent_t = _trace(5, 5, 10, 5, net="INTERCONNECT")
    parent_v = _via(7.5, 5.0, net="INTERCONNECT")

    manifest = build_copper_manifest([child], parent_traces=[parent_t], parent_vias=[parent_v])

    assert manifest.parent_interconnect_traces == 1
    assert manifest.parent_interconnect_vias == 1
    assert manifest.parent_interconnect_length_mm == pytest.approx(5.0)
    assert manifest.total_traces == 2  # 1 child + 1 parent
    assert manifest.total_vias == 1   # 0 child + 1 parent


# ---------------------------------------------------------------------------
# verify_copper_preservation -- perfect preservation
# ---------------------------------------------------------------------------


def test_verify_perfect_preservation():
    """All child traces and vias found in post-route copper."""
    t1 = _trace(0, 0, 5, 0)
    t2 = _trace(5, 0, 5, 3)
    v1 = _via(5, 0)
    child = MockComposedChild("/c", "CHILD", traces=[t1, t2], vias=[v1])
    manifest = build_copper_manifest([child])

    result = verify_copper_preservation(manifest, [t1, t2], [v1])

    assert result["status"] == "PASS"
    assert result["trace_preservation_rate"] == pytest.approx(1.0)
    assert result["via_preservation_rate"] == pytest.approx(1.0)
    assert result["matched_child_traces"] == 2
    assert result["matched_child_vias"] == 1
    assert result["issues"] == []


def test_verify_perfect_with_multiple_children():
    t_a = _trace(0, 0, 3, 0)
    t_b = _trace(10, 0, 13, 0)
    v_b = _via(11.5, 0)
    child_a = MockComposedChild("/a", "A", traces=[t_a], vias=[])
    child_b = MockComposedChild("/b", "B", traces=[t_b], vias=[v_b])
    manifest = build_copper_manifest([child_a, child_b])

    result = verify_copper_preservation(manifest, [t_a, t_b], [v_b])

    assert result["status"] == "PASS"
    assert result["trace_preservation_rate"] == pytest.approx(1.0)
    assert result["per_child"]["/a"]["trace_preservation"] == pytest.approx(1.0)
    assert result["per_child"]["/b"]["trace_preservation"] == pytest.approx(1.0)
    assert result["per_child"]["/b"]["via_preservation"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# verify_copper_preservation -- partial preservation
# ---------------------------------------------------------------------------


def test_verify_partial_preservation():
    """One trace lost, triggers FAIL status when preservation <= 0.95."""
    t1 = _trace(0, 0, 5, 0)
    t2 = _trace(5, 0, 10, 0)
    t3 = _trace(10, 0, 15, 0)
    child = MockComposedChild("/c", "CHILD", traces=[t1, t2, t3], vias=[])
    manifest = build_copper_manifest([child])

    # Only t1 and t2 survive -- t3 is lost
    result = verify_copper_preservation(manifest, [t1, t2], [])

    assert result["status"] == "FAIL"
    assert result["trace_preservation_rate"] == pytest.approx(2 / 3, abs=0.001)
    assert result["matched_child_traces"] == 2
    assert result["expected_child_traces"] == 3
    assert len(result["issues"]) == 1
    assert "lost 1/3 traces" in result["issues"][0]


def test_verify_warn_status_above_95_percent():
    """Loss is small enough (> 95%) that status is WARN, not FAIL."""
    # 40 traces, lose 1 => 39/40 = 0.975 which is strictly > 0.95 => WARN
    traces = [_trace(0, 0, float(i), 0) for i in range(1, 41)]
    child = MockComposedChild("/c", "C", traces=traces, vias=[])
    manifest = build_copper_manifest([child])

    # Drop the last trace
    result = verify_copper_preservation(manifest, traces[:39], [])

    assert result["trace_preservation_rate"] == pytest.approx(39 / 40)
    assert result["status"] == "WARN"


# ---------------------------------------------------------------------------
# verify_copper_preservation -- zero child traces (trivial case)
# ---------------------------------------------------------------------------


def test_verify_zero_child_traces():
    """With no child copper, preservation is trivially 1.0 / PASS."""
    child = MockComposedChild("/empty", "EMPTY", traces=[], vias=[])
    manifest = build_copper_manifest([child])

    result = verify_copper_preservation(manifest, [], [])

    assert result["status"] == "PASS"
    assert result["trace_preservation_rate"] == pytest.approx(1.0)
    assert result["via_preservation_rate"] == pytest.approx(1.0)
    assert result["matched_child_traces"] == 0
    assert result["expected_child_traces"] == 0
    assert result["issues"] == []


def test_verify_empty_manifest():
    """Completely empty manifest (no children at all)."""
    manifest = CopperManifest()

    result = verify_copper_preservation(manifest, [], [])

    assert result["status"] == "PASS"
    assert result["trace_preservation_rate"] == pytest.approx(1.0)
    assert result["via_preservation_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# verify_copper_preservation -- new parent traces added
# ---------------------------------------------------------------------------


def test_verify_new_parent_traces_counted():
    """Extra traces from parent routing appear as new_route_traces."""
    t_child = _trace(0, 0, 5, 0)
    child = MockComposedChild("/c", "CHILD", traces=[t_child], vias=[])
    manifest = build_copper_manifest([child])

    t_new = _trace(20, 20, 30, 20, net="INTERCONNECT")
    v_new = _via(25, 20, net="INTERCONNECT")

    result = verify_copper_preservation(manifest, [t_child, t_new], [v_new])

    assert result["status"] == "PASS"
    assert result["trace_preservation_rate"] == pytest.approx(1.0)
    assert result["new_route_traces"] == 1
    assert result["new_route_vias"] == 1
    assert result["post_route_total_traces"] == 2
    assert result["post_route_total_vias"] == 1


def test_verify_many_new_parent_traces():
    child = MockComposedChild("/c", "C", traces=[_trace(0, 0, 1, 0)], vias=[])
    manifest = build_copper_manifest([child])

    post_traces = [_trace(0, 0, 1, 0)]  # child trace preserved
    for i in range(5):
        post_traces.append(_trace(float(i * 10), 10, float(i * 10 + 5), 10, net="ROUTE"))

    result = verify_copper_preservation(manifest, post_traces, [])

    assert result["status"] == "PASS"
    assert result["new_route_traces"] == 5
    assert result["matched_child_traces"] == 1


# ---------------------------------------------------------------------------
# CopperManifest.to_dict() serialization
# ---------------------------------------------------------------------------


def test_manifest_to_dict_structure():
    t = _trace(0, 0, 3, 4, net="A")
    v = _via(1.5, 2.0, net="A")
    child = MockComposedChild("/child", "MY_CHILD", traces=[t], vias=[v])
    parent_t = _trace(10, 10, 20, 10, net="IC")

    manifest = build_copper_manifest([child], parent_traces=[parent_t])
    d = manifest.to_dict()

    # Top-level keys
    assert "per_child" in d
    assert "total_child_traces" in d
    assert "total_child_vias" in d
    assert "total_child_length_mm" in d
    assert "parent_interconnect_traces" in d
    assert "parent_interconnect_vias" in d
    assert "parent_interconnect_length_mm" in d
    assert "total_traces" in d
    assert "total_vias" in d

    # Values
    assert d["total_child_traces"] == 1
    assert d["total_child_vias"] == 1
    assert d["total_child_length_mm"] == pytest.approx(5.0)
    assert d["parent_interconnect_traces"] == 1
    assert d["parent_interconnect_length_mm"] == pytest.approx(10.0)
    assert d["total_traces"] == 2
    assert d["total_vias"] == 1


def test_manifest_to_dict_per_child_entry():
    t = _trace(0, 0, 6, 8, net="SIG")
    child = MockComposedChild("/leaf", "LEAF_CHILD", traces=[t], vias=[])
    manifest = build_copper_manifest([child])
    d = manifest.to_dict()

    child_d = d["per_child"]["/leaf"]
    assert child_d["instance_path"] == "/leaf"
    assert child_d["sheet_name"] == "LEAF_CHILD"
    assert child_d["trace_count"] == 1
    assert child_d["via_count"] == 0
    assert child_d["total_length_mm"] == pytest.approx(10.0)
    # Fingerprint lists are omitted in serialization for brevity
    assert "trace_fingerprints" not in child_d
    assert "via_fingerprints" not in child_d


def test_manifest_to_dict_rounds_lengths():
    """Length values are rounded to 3 decimal places."""
    # sqrt(2) = 1.41421356...
    t = _trace(0, 0, 1, 1, net="X")
    child = MockComposedChild("/r", "R", traces=[t], vias=[])
    manifest = build_copper_manifest([child])
    d = manifest.to_dict()

    assert d["total_child_length_mm"] == pytest.approx(1.414, abs=0.001)
    assert isinstance(d["total_child_length_mm"], float)
