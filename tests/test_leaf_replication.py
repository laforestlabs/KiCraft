"""Guards for identical-leaf reuse (docs/plans/identical-leaf-reuse-plan.md).

``plan_leaf_replication`` must group structurally-identical repeated channels
(OPTO CH1/CH2 differ only in refs + channel-numbered nets) and yield each
sibling's ``(ref_map, net_map)``, while leaving a topologically-different leaf as
its own representative. ``materialize_sibling`` must write a sibling artifact dir
whose solved_layout carries the sibling's refs/nets and whose mini_pcb carries
the sibling's refdes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kicraft.autoplacer.brain.types import Component, Layer, Pad, Point
from kicraft.cli._leaf_replication import (
    finalize_leaf_replication,
    materialize_sibling,
    plan_leaf_replication,
)


def _pad(ref, pad_id, net):
    return Pad(
        ref=ref,
        pad_id=pad_id,
        pos=Point(0.0, 0.0),
        net=net,
        layer=Layer.FRONT,
        size_mm=Point(1.0, 1.0),
    )


def _comp(ref, value, pads, w=5.0, h=5.0):
    return Component(
        ref=ref,
        value=value,
        pos=Point(0.0, 0.0),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=w,
        height_mm=h,
        pads=pads,
    )


def _channel(u_ref, r_ref, in_net, mid_net):
    """One opto channel: U (opto) -- mid_net -- R (resistor) -- GND; U.1 = input."""
    return {
        u_ref: _comp(u_ref, "OPTO", [_pad(u_ref, "1", in_net), _pad(u_ref, "2", mid_net)]),
        r_ref: _comp(r_ref, "1k", [_pad(r_ref, "1", mid_net), _pad(r_ref, "2", "GND")], w=2.0, h=1.0),
    }


def _node(sheet, path, refs):
    return SimpleNamespace(
        id=SimpleNamespace(
            sheet_name=sheet,
            sheet_file=f"{sheet}.kicad_sch",
            instance_path=path,
            parent_instance_path="/",
        ),
        definition=SimpleNamespace(component_refs=list(refs)),
    )


def _board_state(*channel_dicts):
    components: dict[str, Component] = {}
    for ch in channel_dicts:
        components.update(ch)
    return SimpleNamespace(components=components)


def test_isomorphic_channels_group_with_correct_maps():
    ch1 = _channel("U1", "R1", "IN1", "OPTO_LED1")
    ch2 = _channel("U2", "R2", "IN2", "OPTO_LED2")
    # A structurally-different leaf (3 components) must NOT group.
    diff = {
        "U9": _comp("U9", "MCU", [_pad("U9", "1", "IN9"), _pad("U9", "2", "N")]),
        "R9": _comp("R9", "1k", [_pad("R9", "1", "N"), _pad("R9", "2", "GND")], w=2.0, h=1.0),
        "C9": _comp("C9", "100n", [_pad("C9", "1", "IN9"), _pad("C9", "2", "GND")], w=2.0, h=1.0),
    }
    leaves = [
        _node("OPTO_CH1", "/o1", ["U1", "R1"]),
        _node("OPTO_CH2", "/o2", ["U2", "R2"]),
        _node("MISC", "/m", ["U9", "R9", "C9"]),
    ]
    board = _board_state(ch1, ch2, diff)

    groups = plan_leaf_replication(leaves, board, cfg={})

    assert len(groups) == 2
    g0, g1 = groups
    assert g0.representative.id.instance_path == "/o1"
    assert len(g0.members) == 1
    sib, ref_map, net_map = g0.members[0]
    assert sib.id.instance_path == "/o2"
    assert ref_map == {"U1": "U2", "R1": "R2"}
    # Per-instance signals map to their sibling; the shared rail maps to itself.
    assert net_map["IN1"] == "IN2"
    assert net_map["OPTO_LED1"] == "OPTO_LED2"
    assert net_map["GND"] == "GND"
    # The topologically-different leaf is its own representative, no members.
    assert g1.representative.id.instance_path == "/m"
    assert g1.members == []


def test_kill_switch_disables_grouping():
    ch1 = _channel("U1", "R1", "IN1", "OPTO_LED1")
    ch2 = _channel("U2", "R2", "IN2", "OPTO_LED2")
    leaves = [_node("OPTO_CH1", "/o1", ["U1", "R1"]), _node("OPTO_CH2", "/o2", ["U2", "R2"])]
    board = _board_state(ch1, ch2)

    groups = plan_leaf_replication(leaves, board, cfg={"leaf_replication": False})

    assert len(groups) == 2
    assert all(g.members == [] for g in groups)


def test_materialize_sibling_writes_remapped_artifacts(tmp_path: Path):
    # Minimal representative artifact dir on disk.
    rep_dir = tmp_path / ".experiments" / "subcircuits" / "rep"
    rep_dir.mkdir(parents=True)
    rep_solved_layout = {
        "instance_path": "/o1",
        "components": {
            "U1": {"ref": "U1", "pads": [{"ref": "U1", "pad_id": "1", "net": "IN1"}]},
            "R1": {"ref": "R1", "pads": [{"ref": "R1", "pad_id": "1", "net": "OPTO_LED1"}]},
        },
        "traces": [{"net": "OPTO_LED1"}],
        "vias": [],
        "validation": {"accepted": True},
    }
    (rep_dir / "solved_layout.json").write_text(json.dumps(rep_solved_layout))
    (rep_dir / "debug.json").write_text(json.dumps({"instance_path": "/o1"}))
    (rep_dir / "layout.kicad_pcb").write_text(
        '(footprint "x" (property "Reference" "U1"))\n'
        '(footprint "y" (property "Reference" "R1"))\n'
    )
    rep_metadata = {
        "instance_path": "/o1",
        "sheet_name": "OPTO_CH1",
        "project_dir": str(tmp_path),
        "artifact_paths": {
            "artifact_dir": str(rep_dir),
            "metadata_json": str(rep_dir / "metadata.json"),
            "debug_json": str(rep_dir / "debug.json"),
            "solved_layout_json": str(rep_dir / "solved_layout.json"),
            "mini_pcb": str(rep_dir / "layout.kicad_pcb"),
        },
    }

    sib_node = _node("OPTO_CH2", "/o2", ["U2", "R2"])
    ref_map = {"U1": "U2", "R1": "R2"}
    net_map = {"IN1": "IN2", "OPTO_LED1": "OPTO_LED2"}

    sib_meta = materialize_sibling(rep_metadata, sib_node, ref_map, net_map)

    sib_dir = Path(sib_meta["artifact_paths"]["artifact_dir"])
    assert sib_dir != rep_dir  # distinct dir per instance
    sib_layout = json.loads((sib_dir / "solved_layout.json").read_text())
    # geometry reused, refs + nets remapped, accepted inherited
    assert set(sib_layout["components"]) == {"U2", "R2"}
    assert sib_layout["components"]["U2"]["pads"][0]["net"] == "IN2"
    assert sib_layout["traces"][0]["net"] == "OPTO_LED2"
    assert sib_layout["validation"]["accepted"] is True
    assert sib_layout["instance_path"] == "/o2"
    # identity fields on the sibling metadata
    assert sib_meta["instance_path"] == "/o2"
    assert sib_meta["replicated_from"] == "/o1"
    # maps stored for the post-pin finalize
    assert sib_meta["replication_ref_map"] == ref_map
    assert sib_meta["replication_net_map"] == net_map
    # mini_pcb ref-remapped (nets untouched -- never read from the blocker board)
    sib_pcb = (sib_dir / "layout.kicad_pcb").read_text()
    assert '"U2"' in sib_pcb and '"R2"' in sib_pcb
    assert '"U1"' not in sib_pcb and '"R1"' not in sib_pcb
    # required-by-compose files all present
    assert (sib_dir / "metadata.json").exists()
    assert (sib_dir / "debug.json").exists()


def test_finalize_refreshes_sibling_from_pinned_representative(tmp_path: Path):
    # The rep's PINNED solved_layout picks rotation=180 (its best round); the
    # sibling was left STALE at rotation=0 (a different round). finalize must
    # re-derive the sibling from the rep's pinned geometry.
    root = tmp_path / ".experiments" / "subcircuits"
    rep_dir = root / "rep"
    sib_dir = root / "sib"
    rep_dir.mkdir(parents=True)
    sib_dir.mkdir(parents=True)

    rep_solved = {
        "instance_path": "/o1",
        "components": {
            "U1": {"ref": "U1", "rotation": 180.0, "pads": [{"ref": "U1", "pad_id": "1", "net": "IN1"}]}
        },
        "traces": [{"net": "IN1"}],
    }
    (rep_dir / "solved_layout.json").write_text(json.dumps(rep_solved))
    (rep_dir / "metadata.json").write_text(
        json.dumps({"instance_path": "/o1", "artifact_paths": {"mini_pcb": ""}})
    )

    stale_sib = {
        "instance_path": "/o2",
        "replicated_from": "/o1",
        "components": {
            "U2": {"ref": "U2", "rotation": 0.0, "pads": [{"ref": "U2", "pad_id": "1", "net": "IN2"}]}
        },
        "traces": [{"net": "IN2"}],
    }
    (sib_dir / "solved_layout.json").write_text(json.dumps(stale_sib))
    (sib_dir / "metadata.json").write_text(
        json.dumps(
            {
                "instance_path": "/o2",
                "sheet_name": "OPTO_CH2",
                "sheet_file": "OPTO_CH2.kicad_sch",
                "parent_instance_path": "/",
                "subcircuit_id": {"instance_path": "/o2"},
                "replicated_from": "/o1",
                "replication_ref_map": {"U1": "U2"},
                "replication_net_map": {"IN1": "IN2"},
                "artifact_paths": {"mini_pcb": ""},
            }
        )
    )

    n = finalize_leaf_replication(tmp_path)

    assert n == 1
    refreshed = json.loads((sib_dir / "solved_layout.json").read_text())
    # sibling now carries the rep's PINNED geometry, with the sibling's refs/nets
    assert refreshed["components"]["U2"]["rotation"] == 180.0
    assert refreshed["components"]["U2"]["pads"][0]["net"] == "IN2"
    assert refreshed["traces"][0]["net"] == "IN2"
    assert refreshed["instance_path"] == "/o2"
