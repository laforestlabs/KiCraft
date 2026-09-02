from types import SimpleNamespace

import pytest
from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.leaf_geometry import repair_leaf_placement_legality
from kicraft.autoplacer.brain.leaf_routing import _place_fabricated_edge_interfaces
from kicraft.autoplacer.brain.types import BoardState, Component, Layer, Pad, Point

from kicraft.design.models import BOM, RecipeSelection
from kicraft.design.recipes import expand_recipe, expand_selections
from kicraft.design.recipes.registry import locked_pin_assignments
from kicraft.design.synthesis.validation import check_mcu_programming_access, check_net_coverage
from kicraft.server.stage_contracts import _normalize_stage_response


def _selection(**parameters):
    return RecipeSelection(
        recipe="rp2040-minimal@1",
        instance="main",
        sheets={"mcu": "MCU", "io": "CASTELLATED IO"},
        parameters=parameters,
    )


def test_recipe_expansion_is_deterministic_and_provenanced():
    first = expand_recipe(_selection())
    second = expand_recipe(_selection())
    assert first == second
    assert all(part.recipe_id == "rp2040-minimal@1" for part in first.parts)
    assert len([part for part in first.parts if not part.assembly]) == 34
    assert [edge.side for edge in first.edge_interfaces] == ["left", "right"]

    assert len(first.connections) >= 35
    bom = BOM(
        parts=first.parts,
        connections=first.connections,
        no_connect_pins=first.no_connect_pins,
        edge_interfaces=first.edge_interfaces,
    )
    assert check_net_coverage(bom).ok
    assert check_mcu_programming_access(bom).ok


def test_multiple_recipe_instances_allocate_disjoint_references():
    second = _selection().model_copy(update={"instance": "aux"})
    expansions = expand_selections([_selection(), second])
    refs = [[part.ref for part in expansion.parts] for expansion in expansions]
    assert set(refs[0]).isdisjoint(refs[1])


def test_castellation_solver_pins_pad_centers_to_declared_edge_and_pitch():
    components = {}
    for ref in ("TP1", "TP2", "TP3"):
        components[ref] = Component(
            ref=ref,
            value="Castellated pad",
            pos=Point(5.0, 5.0),
            rotation=0.0,
            layer=Layer.FRONT,
            width_mm=2.2,
            height_mm=2.2,
            pads=[Pad(ref=ref, pad_id="1", pos=Point(5.0, 5.0), net="N", layer=Layer.FRONT)],
        )
    state = BoardState(components=components, nets={})
    state.board_outline = (Point(0.0, 0.0), Point(20.0, 20.0))
    solver = PlacementSolver(
        state,
        {
            "edge_interfaces": [
                {
                    "name": "left-bank",
                    "refs": list(components),
                    "side": "left",
                    "pitch_mm": 2.54,
                }
            ]
        },
        seed=0,
    )
    placed = solver.solve(max_iterations=10)
    centers = [placed[ref].pads[0].pos for ref in components]
    assert [center.x for center in centers] == [0.0, 0.0, 0.0]
    assert [centers[i + 1].y - centers[i].y for i in range(2)] == pytest.approx([2.54, 2.54])
    assert all(component.locked for component in placed.values())


def test_leaf_legality_repair_preserves_solver_owned_castellation_datums():
    components = {
        ref: Component(
            ref=ref,
            value="Castellated pad",
            pos=Point(0.0, 5.0 + index * 2.54),
            rotation=0.0,
            layer=Layer.FRONT,
            width_mm=2.2,
            height_mm=2.2,
            pads=[
                Pad(
                    ref=ref,
                    pad_id="1",
                    pos=Point(0.0, 5.0 + index * 2.54),
                    net=f"N{index}",
                    layer=Layer.FRONT,
                )
            ],
        )
        for index, ref in enumerate(("TP1", "TP2", "TP3"))
    }
    state = BoardState(components=components, nets={})
    state.board_outline = (Point(0.0, 0.0), Point(20.0, 20.0))
    cfg = {
        "edge_interfaces": [
            {
                "name": "left-bank",
                "refs": list(components),
                "side": "left",
                "pitch_mm": 2.54,
            }
        ]
    }

    repaired, _diagnostics = repair_leaf_placement_legality(
        SimpleNamespace(local_state=state),
        components,
        cfg,
    )

    centers = [repaired[ref].pads[0].pos for ref in components]
    assert [center.x for center in centers] == [0.0, 0.0, 0.0]
    assert [centers[i + 1].y - centers[i].y for i in range(2)] == pytest.approx([2.54, 2.54])


def test_trivial_leaf_reframe_reapplies_exact_castellation_datums():
    components = {
        ref: Component(
            ref=ref,
            value="Castellated pad",
            pos=Point(4.0, 3.0 + index * 2.86),
            rotation=0.0,
            layer=Layer.FRONT,
            width_mm=2.2,
            height_mm=2.2,
            pads=[
                Pad(
                    ref=ref,
                    pad_id="1",
                    pos=Point(4.0, 3.0 + index * 2.86),
                    net=f"N{index}",
                    layer=Layer.FRONT,
                )
            ],
        )
        for index, ref in enumerate(("TP1", "TP2", "TP3"))
    }
    state = BoardState(components=components, nets={})
    state.board_outline = (Point(1.0, 1.0), Point(20.0, 20.0))
    _place_fabricated_edge_interfaces(
        state,
        {
            "edge_interfaces": [
                {
                    "refs": list(components),
                    "side": "left",
                    "pitch_mm": 2.54,
                }
            ]
        },
    )
    centers = [components[ref].pads[0].pos for ref in components]
    assert [center.x for center in centers] == [1.0, 1.0, 1.0]
    assert [centers[i + 1].y - centers[i].y for i in range(2)] == pytest.approx([2.54, 2.54])


def test_recipe_rejects_unknown_version_parameter_and_sheet_role():
    with pytest.raises(ValueError, match="unknown circuit recipe"):
        expand_recipe(_selection().model_copy(update={"recipe": "rp2040-minimal@2"}))
    with pytest.raises(ValueError, match="unknown parameters"):
        expand_recipe(_selection(arbitrary_part="x"))
    with pytest.raises(ValueError, match="sheet roles mismatch"):
        expand_recipe(_selection().model_copy(update={"sheets": {"mcu": "MCU"}}))


def test_recipe_locked_wiring_rejects_model_overwrite():
    expansion = expand_recipe(_selection())
    bom = {"parts": [part.model_dump() for part in expansion.parts]}
    locked = locked_pin_assignments(bom)
    (ref, pin), net = next(iter(locked.items()))
    with pytest.raises(Exception, match="recipe-owned"):
        _normalize_stage_response(
            "wiring", {"pins": [{"ref": ref, "pin": pin, "net": net}]}, {"bom": bom}
        )
    merged = _normalize_stage_response("wiring", {"pins": []}, {"bom": bom})[0]
    assert {"ref": "U1", "pin": "26"} in merged["no_connect_pins"]


def test_compact_architecture_range_expands_without_downstream_shape():
    payload = {
        "topologies": {},
        "rail_voltages": {},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "controller"},
            {"name": "CASTELLATED IO", "stem": "CASTELLATED_IO", "function": "edge IO"},
        ],
        "power_nets": [],
        "inter_sheet_nets": [],
        "inter_sheet_net_ranges": [
            {
                "name_pattern": "GPIO{n}",
                "start": 0,
                "end": 29,
                "endpoints": [
                    {"sheet": "MCU", "direction": "bidirectional"},
                    {"sheet": "CASTELLATED IO", "direction": "bidirectional"},
                ],
            }
        ],
    }
    canonical, expanded = _normalize_stage_response("architecture", payload, {})
    assert expanded == 30
    assert [net["name"] for net in canonical["inter_sheet_nets"]] == [f"GPIO{i}" for i in range(30)]
    assert "inter_sheet_net_ranges" not in canonical
