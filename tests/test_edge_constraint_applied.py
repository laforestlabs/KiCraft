"""Regression: a connector with an `edge` component_zone lands on the parent edge.

This is the guard that would have caught a floating USB-C: given
``component_zones = {J1: {edge: bottom}}``, composing the connector's leaf into
a parent must pin J1's pads to the parent's bottom outline (within the
configured inset), not pack it arbitrarily beside the core block.

Runs end-to-end through ``_compose_artifacts`` against synthetic artifacts (no
PCB on disk), exercising the real derive_attachment_constraints -> child_specs
-> placement path.
"""
from __future__ import annotations

from kicraft.autoplacer.brain.subcircuit_instances import (
    LoadedSubcircuitArtifact,
    transform_loaded_artifact,
    transformed_component_map,
)
from kicraft.autoplacer.brain.types import (
    Component,
    Layer,
    Pad,
    Point,
    SubCircuitId,
    SubCircuitLayout,
)
from kicraft.cli.compose_subcircuits import _compose_artifacts


def _id(name: str) -> SubCircuitId:
    return SubCircuitId(
        sheet_name=name,
        sheet_file=f"{name.lower()}.kicad_sch",
        instance_path=f"/{name.lower()}",
    )


def _pad(ref, pad_id, x, y):
    return Pad(ref=ref, pad_id=pad_id, pos=Point(x, y), net="", layer=Layer.FRONT)


def _artifact(name, w, h, components):
    layout = SubCircuitLayout(
        subcircuit_id=_id(name),
        components=dict(components),
        traces=[],
        vias=[],
        bounding_box=(w, h),
        ports=[],
        interface_anchors=[],
        score=75.0,
    )
    return LoadedSubcircuitArtifact(
        artifact_dir=f"/fake/{name}",
        metadata={},
        debug={},
        layout=layout,
        source_files={},
    )


def test_edge_connector_pinned_to_parent_bottom_outline():
    # Core block: an IC roughly centered in a 20x16 leaf.
    core = _artifact(
        "CORE",
        20.0,
        16.0,
        {
            "U2": Component(
                ref="U2",
                value="",
                pos=Point(10.0, 8.0),
                rotation=0.0,
                layer=Layer.FRONT,
                width_mm=8.0,
                height_mm=8.0,
                kind="ic",
                pads=[_pad("U2", "1", 7.0, 8.0), _pad("U2", "2", 13.0, 8.0)],
            )
        },
    )
    # Connector block: J1 with pads at the leaf-local bottom edge (max y).
    mcu = _artifact(
        "MCU",
        12.0,
        6.0,
        {
            "J1": Component(
                ref="J1",
                value="USB-C",
                pos=Point(6.0, 5.0),
                rotation=0.0,
                layer=Layer.FRONT,
                width_mm=9.0,
                height_mm=3.0,
                kind="connector",
                body_center=Point(6.0, 5.0),
                opening_direction=180.0,  # opening faces +y (bottom) in local frame
                pads=[_pad("J1", "A1", 2.0, 5.5), _pad("J1", "A2", 10.0, 5.5)],
            )
        },
    )

    cfg = {
        "component_zones": {"J1": {"edge": "bottom"}},
        "connector_edge_inset_mm": 0.0,  # flush
    }
    state, _ = _compose_artifacts(
        [core, mcu],
        spacing_mm=2.0,
        rotation_step_deg=0.0,
        parent_definition=None,
        pcb_path=None,
        cfg=cfg,
        seed=0,
    )

    parent_bottom = state.bounding_box[1].y
    # locate the MCU entry and transform J1 into the parent frame
    mcu_entry = next(e for e in state.entries if e.sheet_name == "MCU")
    core_entry = next(e for e in state.entries if e.sheet_name == "CORE")
    j1 = transformed_component_map(
        transform_loaded_artifact(mcu, mcu_entry.origin, mcu_entry.rotation)
    )["J1"]
    u2 = transformed_component_map(
        transform_loaded_artifact(core, core_entry.origin, core_entry.rotation)
    )["U2"]

    j1_pad_max_y = max(p.pos.y for p in j1.pads)
    # J1's outboard pads must sit at the parent's bottom outline (within a small
    # tolerance for inset/rounding), and below the core IC.
    assert abs(j1_pad_max_y - parent_bottom) <= 1.5, (
        f"J1 pads (max y={j1_pad_max_y:.2f}) not pinned to parent bottom "
        f"outline (y={parent_bottom:.2f}); edge constraint not applied"
    )
    assert j1_pad_max_y > u2.pos.y, "edge connector must be below the core block"
