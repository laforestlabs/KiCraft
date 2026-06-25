"""Pure-function tests for the conceptual stage diagram builders.

Exercises both adapters against the committed BMP280 fixture plus synthetic
edge cases. No NiceGUI, no network -- the builders are plain data-in/option-out.
"""
from __future__ import annotations

import json
import copy
from pathlib import Path

from kicraft.server import stage_diagram as sd

FIXTURE = Path(__file__).parent / "fixtures" / "bmp280_reader_state.json"


def _load() -> dict:
    return json.loads(FIXTURE.read_text())


def _series(opt: dict) -> dict:
    return opt["series"][0]


# ----------------------------------------------------------- functional_spec

def test_functional_spec_none_on_empty_slot():
    assert sd.functional_spec_diagram({}) is None
    assert sd.functional_spec_diagram({"blocks": []}) is None


def test_functional_spec_bmp280_nodes_and_categories():
    st = _load()
    opt = sd.functional_spec_diagram(st["functional_spec"])
    assert opt is not None
    data = _series(opt)["data"]
    names = [n["name"] for n in data]
    # The five BMP280 blocks, in input order.
    assert names == ["USB_INPUT", "LDO_3V3", "MCU", "SENSOR", "USER_IO"]
    # Category indices map onto the legend categories actually present.
    cats = _series(opt)["categories"]
    cat_names = [c["name"] for c in cats]
    assert set(cat_names) == {"interface", "power", "process", "sense"}
    # power column is leftmost -> smallest x.
    by_name = {n["name"]: n for n in data}
    assert by_name["LDO_3V3"]["x"] == 0  # power -> column 0
    # purpose surfaces as the tooltip value.
    assert "BMP280" in by_name["SENSOR"]["value"]


def test_functional_spec_directed_edges_typed_by_signal():
    st = _load()
    links = _series(sd.functional_spec_diagram(st["functional_spec"]))["links"]
    # USB_INPUT -> LDO_3V3 appears twice (power + ground); find the power one.
    power_edge = next(
        e for e in links
        if e["source"] == "USB_INPUT" and e["target"] == "LDO_3V3"
        and e["lineStyle"]["color"] == sd.SIGNAL_LINE_STYLE["power"]["color"]
    )
    assert power_edge["edgeSymbol"] == ["none", "arrow"]
    # The I2C bus edge is dashed.
    bus = next(e for e in links if e["source"] == "MCU" and e["target"] == "SENSOR")
    assert bus["lineStyle"].get("type") == "dashed"


def test_functional_spec_nodes_only_when_no_connections():
    slot = {"blocks": [{"name": "A", "category": "power", "purpose": "p"}],
            "connections": []}
    opt = sd.functional_spec_diagram(slot)
    assert opt is not None
    assert _series(opt)["links"] == []
    assert len(_series(opt)["data"]) == 1


def test_functional_spec_count_badge():
    slot = {"blocks": [{"name": "LED", "category": "drive", "purpose": "x", "count": 9}],
            "connections": []}
    node = _series(sd.functional_spec_diagram(slot))["data"][0]
    assert node["name"] == "LED ×9"


def test_functional_spec_deterministic():
    st = _load()
    a = sd.functional_spec_diagram(st["functional_spec"])
    b = sd.functional_spec_diagram(copy.deepcopy(st["functional_spec"]))
    assert a == b


# --------------------------------------------------------------- architecture

def test_architecture_none_on_empty_slot():
    assert sd.architecture_diagram({}) is None
    assert sd.architecture_diagram({"sheets": []}) is None


def test_architecture_bmp280_sheet_nodes():
    st = _load()
    opt = sd.architecture_diagram(st["architecture"])
    assert opt is not None
    data = _series(opt)["data"]
    sheet_nodes = [n for n in data if n["symbol"] == "circle"]
    assert len(sheet_nodes) == 5
    assert {n["name"] for n in sheet_nodes} == {
        "USB INPUT", "LDO 3V3", "MCU", "SENSOR", "USER IO"}
    # All sheets share the "Sheet" category (index 0).
    assert all(n["category"] == 0 for n in sheet_nodes)


def test_architecture_power_net_becomes_hub_with_spokes():
    st = _load()
    data = _series(sd.architecture_diagram(st["architecture"]))["data"]
    hubs = [n for n in data if n["symbol"] == "diamond"]
    hub_names = {n["name"] for n in hubs}
    # +3V3 (3 sheets) and the 2-endpoint power nets VBUS/GND all become hubs.
    assert "net:+3V3" in hub_names
    assert "net:VBUS" in hub_names
    assert "net:GND" in hub_names
    # +3V3 hub is a power hub (category 1, amber).
    plus3 = next(n for n in hubs if n["name"] == "net:+3V3")
    assert plus3["category"] == 1
    assert plus3["itemStyle"]["color"] == sd.POWER_HUB_COLOR


def test_architecture_hub_spoke_count_matches_endpoints():
    st = _load()
    opt = sd.architecture_diagram(st["architecture"])
    links = _series(opt)["links"]
    # +3V3 has 3 endpoints (LDO 3V3, MCU, SENSOR) -> 3 undirected spokes.
    plus3_spokes = [e for e in links if e["source"] == "net:+3V3"]
    assert len(plus3_spokes) == 3
    assert all(e["edgeSymbol"] == ["none", "none"] for e in plus3_spokes)
    # GND has 5 endpoints -> 5 spokes.
    gnd_spokes = [e for e in links if e["source"] == "net:GND"]
    assert len(gnd_spokes) == 5


def test_architecture_two_endpoint_signal_net_is_single_edge():
    st = _load()
    links = _series(sd.architecture_diagram(st["architecture"]))["links"]
    # SDA / SCL are 2-endpoint bidirectional signal nets -> one undirected
    # edge each between MCU and SENSOR (no hub node for them).
    sda = [e for e in links
           if e.get("label", {}).get("formatter") == "SDA"]
    assert len(sda) == 1
    assert sda[0]["edgeSymbol"] == ["none", "none"]
    assert {sda[0]["source"], sda[0]["target"]} == {"MCU", "SENSOR"}
    assert "net:SDA" not in {n["name"] for n in _series(sd.architecture_diagram(st["architecture"]))["data"]}


def test_architecture_directed_edge_for_output_input_net():
    st = _load()
    links = _series(sd.architecture_diagram(st["architecture"]))["links"]
    # BTN: USER IO (output) -> MCU (input) -> directed arrow.
    btn = [e for e in links
           if e.get("label", {}).get("formatter") == "BTN"]
    assert len(btn) == 1
    assert btn[0]["edgeSymbol"] == ["none", "arrow"]
    assert btn[0]["source"] == "USER IO"
    assert btn[0]["target"] == "MCU"


def test_architecture_deterministic():
    st = _load()
    a = sd.architecture_diagram(st["architecture"])
    b = sd.architecture_diagram(copy.deepcopy(st["architecture"]))
    assert a == b


# --------------------------------------------------------- replication collapse

def test_architecture_replication_group_collapses():
    slot = {
        "sheets": [
            {"name": "AXIS X", "stem": "AXIS_X", "function": "x",
             "replication_group": "AXIS", "replication_instance": 1},
            {"name": "AXIS Y", "stem": "AXIS_Y", "function": "y",
             "replication_group": "AXIS", "replication_instance": 2},
            {"name": "AXIS Z", "stem": "AXIS_Z", "function": "z",
             "replication_group": "AXIS", "replication_instance": 3},
            {"name": "MCU", "stem": "MCU", "function": "mcu"},
        ],
        "power_nets": [],
        "inter_sheet_nets": [
            # STEP touches all three instances -> collapses to one spoke.
            {"name": "STEP", "endpoints": [
                {"sheet": "MCU", "direction": "output"},
                {"sheet": "AXIS X", "direction": "input"},
                {"sheet": "AXIS Y", "direction": "input"},
                {"sheet": "AXIS Z", "direction": "input"},
            ]},
        ],
    }
    opt = sd.architecture_diagram(slot)
    assert opt is not None
    data = _series(opt)["data"]
    sheet_nodes = [n for n in data if n["symbol"] == "circle"]
    # Three AXIS instances collapse to one representative (×3) + MCU = 2 nodes.
    assert len(sheet_nodes) == 2
    axis = next(n for n in sheet_nodes if "AXIS" in n["name"])
    assert "×3" in axis["name"]
    # STEP touches 3 replicated instances which collapse to one representative,
    # so the net becomes a 2-endpoint directed edge (MCU out -> AXIS rep in)
    # rather than a hub. No net:STEP hub node exists.
    assert "net:STEP" not in {n["name"] for n in data}
    step = [e for e in _series(opt)["links"]
            if e.get("label", {}).get("formatter") == "STEP"]
    assert len(step) == 1
    assert step[0]["edgeSymbol"] == ["none", "arrow"]
    assert step[0]["source"] == "MCU"
    assert "AXIS" in step[0]["target"]


def test_architecture_power_hub_color_for_declared_power_net():
    # A net named "RAIL" that isn't matched by name heuristics but is declared
    # in power_nets should still become a power-coloured hub.
    slot = {
        "sheets": [
            {"name": "A", "stem": "A", "function": "a"},
            {"name": "B", "stem": "B", "function": "b"},
            {"name": "C", "stem": "C", "function": "c"},
        ],
        "power_nets": ["RAIL"],
        "inter_sheet_nets": [
            {"name": "RAIL", "endpoints": [
                {"sheet": "A", "direction": "bidirectional"},
                {"sheet": "B", "direction": "bidirectional"},
                {"sheet": "C", "direction": "bidirectional"},
            ]},
        ],
    }
    data = _series(sd.architecture_diagram(slot))["data"]
    hub = next(n for n in data if n["symbol"] == "diamond")
    assert hub["category"] == 1
    assert hub["itemStyle"]["color"] == sd.POWER_HUB_COLOR


# --------------------------------------------------------- web.py integration

def test_inspector_spec_prepends_graph_section():
    from kicraft.server import web
    st = _load()
    for stage in ("functional_spec", "architecture"):
        secs = web._inspector_spec(stage, st, {}, Path("."), [])
        assert secs, f"{stage} produced no sections"
        assert secs[0]["type"] == "graph"
        assert secs[0]["title"] == "Concept diagram"
        assert "series" in secs[0]["option"]
        # The precise tables still follow the diagram.
        assert any(s["type"] == "table" for s in secs[1:])


def test_inspector_spec_empty_slot_returns_empty():
    from kicraft.server import web
    assert web._inspector_spec("functional_spec", {}, {}, Path("."), []) == []
    assert web._inspector_spec("architecture", {}, {}, Path("."), []) == []
