from types import SimpleNamespace

import pytest
from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.leaf_geometry import repair_leaf_placement_legality
from kicraft.autoplacer.brain.leaf_routing import _place_fabricated_edge_interfaces
from kicraft.autoplacer.brain.types import BoardState, Component, Layer, Pad, Point

from kicraft.design.models import BOM, RecipeSelection
from kicraft.design.recipes import expand_recipe, expand_selections
from kicraft.design.recipes.pin_allocator import PinAllocationError, allocate_pins
from kicraft.design.recipes.registry import get_recipe, locked_pin_assignments
from kicraft.design.synthesis.validation import check_mcu_programming_access, check_net_coverage
from kicraft.server.stage_contracts import StageSchemaError, _normalize_stage_response


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
        expand_recipe(_selection().model_copy(update={"recipe": "rp2040-minimal@3"}))
    with pytest.raises(ValueError, match="unknown parameters"):
        expand_recipe(_selection(arbitrary_part="x"))
    with pytest.raises(ValueError, match="sheet roles mismatch"):
        expand_recipe(_selection().model_copy(update={"sheets": {"mcu": "MCU"}}))


def test_rp2040_v2_scopes_internal_nets_and_binds_power_ports():
    selection = RecipeSelection(
        recipe="rp2040-minimal@2",
        instance="main",
        sheets={"mcu": "MCU", "io": "IO"},
        port_bindings={"vdd": "+3V3", "gnd": "GND"},
        requirement_ids=["mcu_core"],
    )
    first = expand_recipe(selection)
    second = expand_recipe(
        selection.model_copy(update={"instance": "aux", "requirement_ids": ["aux_core"]})
    )
    first_nets = {connection.net_name for connection in first.connections}
    assert "+3V3" in first_nets
    assert "QSPI_CS" not in first_nets
    assert set(first.ownership.internal_nets).isdisjoint(
        second.ownership.internal_nets
    )
    definition = get_recipe("rp2040-minimal@2")
    assert definition.maturity == "production"
    assert len(definition.source_documents) == 2


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


def test_allocator_owned_wiring_pin_rejects_model_overwrite():
    from kicraft.design.models import RecipePinAllocation

    selection = RecipeSelection(
        recipe="esp32-s3-mini-1-minimal@1",
        instance="main",
        sheets={"mcu": "MCU"},
        parameters={"native_usb": False},
        port_bindings={"vdd": "+3V3", "gnd": "GND"},
        requirement_ids=["mcu_core"],
        pin_allocations=[
            RecipePinAllocation(net="APP_OUT", pin="5", capability="output")
        ],
    )
    expansion = expand_recipe(selection)
    bom = {
        "parts": [part.model_dump(mode="json") for part in expansion.parts],
        "connections": [
            connection.model_dump(mode="json") for connection in expansion.connections
        ],
        "no_connect_pins": [
            pin.model_dump(mode="json") for pin in expansion.no_connect_pins
        ],
        "recipe_ownership": [expansion.ownership.model_dump(mode="json")],
    }
    with pytest.raises(StageSchemaError, match="recipe-owned"):
        _normalize_stage_response(
            "wiring",
            {"pins": [{"ref": "U1", "pin": "5", "net": "APP_OUT"}]},
            {"bom": bom},
        )


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


def test_architecture_normalizes_sheet_stems_used_as_endpoint_names():
    payload = {
        "topologies": {},
        "rail_voltages": {},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {
                "name": "DIGITAL_INPUT_HEADER",
                "stem": "DIGITAL_INPUT_HEADER",
                "function": "logic input",
            },
            {"name": "R2R LADDER", "stem": "R2R_LADDER", "function": "DAC ladder"},
        ],
        "power_nets": [],
        "inter_sheet_nets": [
            {
                "name": "D0",
                "endpoints": [
                    {"sheet": "DIGITAL_INPUT_HEADER", "direction": "output"},
                    {"sheet": "R2R_LADDER", "direction": "input"},
                ],
            }
        ],
    }

    canonical, expanded = _normalize_stage_response("architecture", payload, {})

    assert expanded == 0
    assert [sheet["name"] for sheet in canonical["sheets"]] == [
        "DIGITAL INPUT HEADER",
        "R2R LADDER",
    ]
    assert [endpoint["sheet"] for endpoint in canonical["inter_sheet_nets"][0]["endpoints"]] == [
        "DIGITAL INPUT HEADER",
        "R2R LADDER",
    ]


def _range_architecture_payload(explicit_nets, ranges):
    return {
        "topologies": {},
        "rail_voltages": {},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {"name": "MOTOR 1", "stem": "MOTOR1", "function": "motor stage"},
            {"name": "MOTOR 2", "stem": "MOTOR2", "function": "motor stage"},
            {"name": "MCU", "stem": "MCU", "function": "controller"},
        ],
        "power_nets": [],
        "inter_sheet_nets": explicit_nets,
        "inter_sheet_net_ranges": ranges,
    }


_MOTOR1_MCU = [
    {"sheet": "MOTOR 1", "direction": "bidirectional"},
    {"sheet": "MCU", "direction": "bidirectional"},
]


def test_compact_architecture_range_deduplicates_identical_explicit_net():
    # MOTOR1_A is declared explicitly and again through MOTOR{n}_A 1..2 with the
    # same endpoints in reversed order: endpoint order is not semantically
    # meaningful, so the redundant range expansion is dropped losslessly.
    payload = _range_architecture_payload(
        explicit_nets=[{"name": "MOTOR1_A", "endpoints": _MOTOR1_MCU}],
        ranges=[
            {
                "name_pattern": "MOTOR{n}_A",
                "start": 1,
                "end": 2,
                "endpoints": list(reversed(_MOTOR1_MCU)),
            }
        ],
    )
    canonical, expanded = _normalize_stage_response("architecture", payload, {})
    names = [net["name"] for net in canonical["inter_sheet_nets"]]
    assert names.count("MOTOR1_A") == 1
    assert names == ["MOTOR1_A", "MOTOR2_A"]
    assert expanded == 1
    assert "inter_sheet_net_ranges" not in canonical


def test_compact_architecture_range_rejects_conflicting_explicit_net():
    # Same generated name exists explicitly with different endpoint semantics.
    payload = _range_architecture_payload(
        explicit_nets=[
            {
                "name": "MOTOR1_A",
                "endpoints": [
                    {"sheet": "MOTOR 1", "direction": "output"},
                    {"sheet": "MCU", "direction": "input"},
                ],
            }
        ],
        ranges=[
            {
                "name_pattern": "MOTOR{n}_A",
                "start": 1,
                "end": 2,
                "endpoints": _MOTOR1_MCU,
            }
        ],
    )
    with pytest.raises(StageSchemaError, match="duplicate/overlapping inter-sheet net 'MOTOR1_A'"):
        _normalize_stage_response("architecture", payload, {})

    # Same generated name exists explicitly over a different sheet set.
    payload = _range_architecture_payload(
        explicit_nets=[
            {
                "name": "MOTOR1_A",
                "endpoints": [
                    {"sheet": "MOTOR 2", "direction": "bidirectional"},
                    {"sheet": "MCU", "direction": "bidirectional"},
                ],
            }
        ],
        ranges=[
            {
                "name_pattern": "MOTOR{n}_A",
                "start": 1,
                "end": 2,
                "endpoints": _MOTOR1_MCU,
            }
        ],
    )
    with pytest.raises(StageSchemaError, match="duplicate/overlapping inter-sheet net 'MOTOR1_A'"):
        _normalize_stage_response("architecture", payload, {})

    # A second range covering an already generated name stays rejected even
    # when both ranges carry identical endpoints.
    payload = _range_architecture_payload(
        explicit_nets=[],
        ranges=[
            {"name_pattern": "GPIO{n}", "start": 0, "end": 4, "endpoints": _MOTOR1_MCU},
            {"name_pattern": "GPIO{n}", "start": 3, "end": 9, "endpoints": _MOTOR1_MCU},
        ],
    )
    with pytest.raises(StageSchemaError, match="duplicate/overlapping inter-sheet net 'GPIO3'"):
        _normalize_stage_response("architecture", payload, {})


def _esp32_architecture_payload():
    return {
        "topologies": {"MCU": "ESP32-S3 controller with native USB recovery"},
        "rail_voltages": {"+3V3": 3.3},
        "comms_protocols": ["USB"],
        "mcu_present": True,
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "ESP32-S3 controller"},
        ],
        "power_nets": ["+3V3"],
        "inter_sheet_nets": [],
        "assumptions": ["3V3 is supplied externally at 500mA"],
    }


def test_architecture_exact_esp32_part_resolves_without_model_selection():
    canonical, _expanded = _normalize_stage_response(
        "architecture",
        _esp32_architecture_payload(),
        {"intent": {"named_parts": ["ESP32-S3-MINI-1-N8"]}},
    )
    assert canonical["requirements"][0]["exact_part"] == "ESP32-S3-MINI-1-N8"
    selection = canonical["recipe_selections"][0]
    assert selection["recipe"] == "esp32-s3-mini-1-minimal@1"
    assert selection["port_bindings"] == {"gnd": "GND", "vdd": "+3V3"}
    assert canonical["recipe_resolution"][0]["recipe"] == selection["recipe"]


def test_architecture_unsupported_protected_esp32_variant_blocks():
    with pytest.raises(
        StageSchemaError,
        match="unsupported_protected_variant",
    ):
        _normalize_stage_response(
            "architecture",
            _esp32_architecture_payload(),
            {"intent": {"named_parts": ["ESP32-S3-WROOM-1-N16R8"]}},
        )


def _typed_esp32_architecture(**requirement_overrides):
    payload = _esp32_architecture_payload()
    requirement = {
        "id": "mcu_core",
        "sheet": "MCU",
        "role": "mcu_core",
        "family": "esp32-s3-module",
        "exact_part": "ESP32-S3-MINI-1-N8",
        "parameters": {},
        "ports": {"vdd": "+3V3", "gnd": "GND"},
        "interfaces": [],
        **requirement_overrides,
    }
    payload["requirements"] = [requirement]
    return payload


def test_recipe_resolver_blocks_explicit_parameter_and_identity_conflicts():
    with pytest.raises(StageSchemaError, match="conflicting_recipe_parameter"):
        _normalize_stage_response(
            "architecture",
            _typed_esp32_architecture(
                parameters={"native_usb": False},
                interfaces=["usb_device"],
            ),
            {},
        )
    with pytest.raises(StageSchemaError, match="conflicting_exact_variant"):
        _normalize_stage_response(
            "architecture",
            _typed_esp32_architecture(),
            {"intent": {"named_parts": ["RP2040"]}},
        )


def test_recipe_resolver_requires_conditional_usb_ports_and_declared_nets():
    with pytest.raises(StageSchemaError, match="missing_recipe_port"):
        _normalize_stage_response(
            "architecture",
            _typed_esp32_architecture(parameters={"native_usb": True}),
            {},
        )
    with pytest.raises(StageSchemaError, match="unknown_recipe_port_net"):
        _normalize_stage_response(
            "architecture",
            _typed_esp32_architecture(
                ports={
                    "vdd": "+3V3",
                    "gnd": "GND",
                    "usb_dm": "UNDECLARED_DM",
                    "usb_dp": "UNDECLARED_DP",
                }
            ),
            {},
        )


def test_esp32_optional_usb_and_internal_nets_are_instance_scoped():
    base = RecipeSelection(
        recipe="esp32-s3-mini-1-minimal@1",
        instance="main",
        sheets={"mcu": "MCU"},
        parameters={"native_usb": False},
        port_bindings={"vdd": "+3V3", "gnd": "GND"},
        requirement_ids=["mcu_core"],
    )
    without_usb = expand_recipe(base)
    with_usb = expand_recipe(
        base.model_copy(
            update={
                "parameters": {"native_usb": True},
                "port_bindings": {
                    "vdd": "+3V3",
                    "gnd": "GND",
                    "usb_dm": "USB_DM",
                    "usb_dp": "USB_DP",
                },
            }
        )
    )
    second = expand_recipe(
        base.model_copy(update={"instance": "aux", "requirement_ids": ["aux_core"]})
    )
    assert len(with_usb.parts) == len(without_usb.parts) + 2
    assert set(without_usb.ownership.internal_nets).isdisjoint(
        second.ownership.internal_nets
    )
    assert {pin.pin for pin in without_usb.no_connect_pins} >= {"23", "24"}
    assert {connection.net_name for connection in with_usb.connections} >= {
        "USB_DM",
        "USB_DP",
    }


def test_pin_allocator_is_order_stable_and_reports_impossible_parallel_bus():
    from kicraft.design.recipes.pin_allocator import PinAllocationRequest

    pins = get_recipe("esp32-s3-mini-1-minimal@1").allocatable_pins
    requests = [
        PinAllocationRequest(
            id=f"d_{index}",
            net=f"D{index}",
            capability="output",
            group="parallel",
            contiguous=True,
        )
        for index in range(13)
    ]
    first = allocate_pins(pins, requests)
    second = allocate_pins(pins, list(reversed(requests)))
    assert first == second
    pin_by_number = {pin.pin: pin.gpio for pin in pins}
    by_net = {allocation.net: pin_by_number[allocation.pin] for allocation in first}
    assert [by_net[f"D{index}"] for index in range(13)] == list(
        range(by_net["D0"], by_net["D0"] + 13)
    )
    assert allocate_pins(pins, requests, existing=first) == first
    with pytest.raises(PinAllocationError, match="unsatisfied_pin_capability"):
        allocate_pins(
            pins,
            [
                PinAllocationRequest(
                    id=f"d_{index}",
                    net=f"D{index}",
                    capability="output",
                    group="parallel",
                    contiguous=True,
                )
                for index in range(32)
            ],
        )


def test_pin_allocator_excludes_reserved_input_only_and_strapping_pins():
    from kicraft.design.models import RecipePinAllocation
    from kicraft.design.recipes.models import RecipeAllocatablePin
    from kicraft.design.recipes.pin_allocator import PinAllocationRequest

    pins = (
        RecipeAllocatablePin(
            role="mcu", pin="1", gpio=0, capabilities=("gpio", "output"), strapping=True
        ),
        RecipeAllocatablePin(
            role="mcu", pin="2", gpio=1, capabilities=("gpio", "output")
        ),
        RecipeAllocatablePin(
            role="mcu",
            pin="3",
            gpio=2,
            capabilities=("gpio", "input"),
            input_only=True,
        ),
        RecipeAllocatablePin(
            role="mcu", pin="4", gpio=3, capabilities=("gpio", "output"), reserved=True
        ),
    )
    output = PinAllocationRequest(id="out", net="OUT", capability="output")
    assert allocate_pins(pins, [output])[0].pin == "2"
    assert (
        allocate_pins(
            (pins[0],),
            [output.model_copy(update={"allow_strapping": True})],
        )[0].pin
        == "1"
    )
    duplicate_existing = [
        RecipePinAllocation(net="A", pin="2", capability="output"),
        RecipePinAllocation(net="B", pin="2", capability="output"),
    ]
    with pytest.raises(PinAllocationError, match="unsatisfied_pin_capability"):
        allocate_pins(
            pins,
            [
                output.model_copy(update={"id": "a", "net": "A", "group": "bus"}),
                output.model_copy(update={"id": "b", "net": "B", "group": "bus"}),
            ],
            existing=duplicate_existing,
        )


@pytest.mark.parametrize(
    ("interface", "ports", "capabilities"),
    [
        (
            "i2c_controller",
            {"sda": "I2C_SDA", "scl": "I2C_SCL"},
            {"i2c-sda", "i2c-scl"},
        ),
        (
            "spi_controller",
            {"sclk": "SCLK", "mosi": "MOSI", "miso": "MISO", "cs": "CS"},
            {"spi-sclk", "spi-mosi", "spi-miso", "spi-cs"},
        ),
        ("uart", {"tx": "TX", "rx": "RX"}, {"uart-tx", "uart-rx"}),
    ],
)
def test_pin_allocator_allocates_controller_buses_atomically(
    interface, ports, capabilities
):
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins

    requirement = CircuitRequirement(
        id="mcu_core",
        sheet="MCU",
        role="mcu_core",
        family="esp32-s3-module",
        ports=ports,
        interfaces=[interface],
    )
    allocations = allocate_requirement_pins(
        get_recipe("esp32-s3-mini-1-minimal@1"),
        requirement,
    )
    assert {allocation.capability for allocation in allocations} == capabilities
    assert len({allocation.pin for allocation in allocations}) == len(allocations)
