"""build_ratsnest: cross-leaf net links for the manual layout canvas.

Anchors come from each leaf's solved_layout.json interface_anchors
(leaf solver's re-based local frame) and must land in the canvas's
leaf-local page frame via the metadata local_board_outline top-left
offset. Only nets anchored in >= 2 distinct leaves survive; GND is
plane-connected and never drawn.
"""

from __future__ import annotations

import json
from pathlib import Path

from kicraft.layout_editor.leaves import LeafInfo
from kicraft.layout_editor.ratsnest import build_ratsnest


def _leaf(tmp_path: Path, name: str, ip: str, *, anchors, ports=None,
          top_left=(100.0, 50.0)) -> LeafInfo:
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "solved_layout.json").write_text(json.dumps({
        "instance_path": ip,
        "interface_anchors": [
            {"port_name": port, "pos": {"x": x, "y": y}, "layer": "F.Cu"}
            for (port, x, y) in anchors
        ],
    }), encoding="utf-8")
    (d / "metadata.json").write_text(json.dumps({
        "instance_path": ip,
        "interface_ports": [
            {"name": n, "net_name": net} for (n, net) in (ports or [])
        ],
        "local_board_outline": {
            "top_left_x": top_left[0], "top_left_y": top_left[1],
            "width_mm": 20.0, "height_mm": 10.0,
        },
    }), encoding="utf-8")
    return LeafInfo(
        instance_path=ip, sheet_name=name, width_mm=20.0, height_mm=10.0,
        artifact_dir=d,
    )


def test_cross_leaf_net_with_offset_and_exclusions(tmp_path: Path):
    a = _leaf(tmp_path, "A", "/a",
              anchors=[("SIG1", 1.0, 2.0), ("GND", 3.0, 3.0), ("LOCAL", 4.0, 4.0)],
              top_left=(100.0, 50.0))
    b = _leaf(tmp_path, "B", "/b",
              anchors=[("SIG1", 5.0, 6.0)],
              top_left=(200.0, 80.0))

    nets = build_ratsnest([a, b])

    # GND excluded (plane), LOCAL excluded (single leaf), SIG1 kept.
    assert [n["net"] for n in nets] == ["SIG1"]
    anchors = {an["instance_path"]: an for an in nets[0]["anchors"]}
    # Re-based local anchor + the leaf's page-frame top-left offset.
    assert anchors["/a"]["x"] == 101.0 and anchors["/a"]["y"] == 52.0
    assert anchors["/b"]["x"] == 205.0 and anchors["/b"]["y"] == 86.0


def test_port_alias_maps_to_net_and_sibling_fallback(tmp_path: Path):
    # Rep: the port is named differently from its net -- the metadata
    # map must apply. Sibling: anchors already carry the remapped net
    # while the metadata ports are the rep's (stale) -- the port_name
    # fallback must win.
    rep = _leaf(tmp_path, "REP", "/rep",
                anchors=[("COIL_PORT", 1.0, 1.0)],
                ports=[("COIL_PORT", "RELAY_COIL1")])
    sib = _leaf(tmp_path, "SIB", "/sib",
                anchors=[("RELAY_COIL1", 2.0, 2.0)],
                ports=[("OTHER", "OTHER_NET")])

    nets = build_ratsnest([rep, sib])

    assert [n["net"] for n in nets] == ["RELAY_COIL1"]
    assert len(nets[0]["anchors"]) == 2


def test_edge_fallback_when_metadata_outline_missing(tmp_path: Path):
    a = _leaf(tmp_path, "A", "/a", anchors=[("SIG", 1.0, 1.0)])
    b = _leaf(tmp_path, "B", "/b", anchors=[("SIG", 2.0, 2.0)])
    # Strip the outline from A's metadata; its LeafInfo edge AABB must
    # be used as the offset instead.
    meta_path = a.artifact_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    del meta["local_board_outline"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    a.edge_min_x, a.edge_min_y = 10.0, 20.0

    nets = build_ratsnest([a, b])

    anchors = {an["instance_path"]: an for an in nets[0]["anchors"]}
    assert anchors["/a"]["x"] == 11.0 and anchors["/a"]["y"] == 21.0
