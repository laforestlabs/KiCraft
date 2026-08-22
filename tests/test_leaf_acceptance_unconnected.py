"""Tests for the leaf-acceptance unconnected-nets gate.

A leaf must route all of its *signal* nets. Power/ground nets are excluded
because they close on the post-route copper pour, not on leaf routing.
"""

from __future__ import annotations

from kicraft.autoplacer.brain.leaf_acceptance import (
    LeafAcceptanceConfig,
    acceptance_config_from_dict,
    evaluate_leaf_acceptance,
)
from kicraft.autoplacer.routing_board import parse_unconnected_nets


# Real kicad-cli text DRC report excerpt from the #38 USB-POWER leaf.
_REPORT = """\
** Found 6 unconnected pads **
[unconnected_items]: Missing connection between items
    Local override; error
    @(8.1050 mm, 9.7790 mm): Pad A4B9 [VBUS] of J1 on F.Cu
    @(8.1050 mm, 14.5790 mm): Pad B4A9 [VBUS] of J1 on F.Cu
[unconnected_items]: Missing connection between items
    Local override; error
    @(7.3450 mm, 16.5090 mm): PTH pad 1 [GND] of J1
    @(8.1050 mm, 15.3790 mm): Pad B1A12 [GND] of J1 on F.Cu
[unconnected_items]: Missing connection between items
    Local override; error
    @(8.1050 mm, 13.9290 mm): Pad B5 [CC2] of J1 on F.Cu
    @(10.6900 mm, 1.4950 mm): Pad 1 [CC2] of R2 on F.Cu
"""


def _validation(unconnected: int, nets: list[str]) -> dict:
    return {
        "board_exists": True,
        "drc": {"shorts": 0, "unconnected": unconnected, "unconnected_nets": nets},
        "track_summary": {"traces": 10, "vias": 0},
    }


def testparse_unconnected_nets_dedupes_and_orders():
    assert parse_unconnected_nets(_REPORT) == ["VBUS", "GND", "CC2"]


def testparse_unconnected_nets_empty_when_clean():
    assert parse_unconnected_nets("** Found 0 unconnected pads **\n") == []


def test_signal_net_unconnected_is_rejected():
    cfg = LeafAcceptanceConfig(max_unconnected=0)
    res = evaluate_leaf_acceptance(_validation(3, ["VBUS", "GND", "CC2"]), {}, cfg)
    assert res.accepted is False
    assert "no_unconnected" in res.rejection_reasons
    gate = res.gate_results["no_unconnected"]
    assert gate["signal_unconnected_nets"] == ["CC2"]
    assert set(gate["ignored_poured_nets"]) == {"VBUS", "GND"}


def test_power_ground_only_unconnected_is_accepted():
    # GND/VBUS close on the pour, so a leaf with only those unrouted passes.
    cfg = LeafAcceptanceConfig(max_unconnected=0)
    res = evaluate_leaf_acceptance(_validation(2, ["VBUS", "GND"]), {}, cfg)
    assert res.accepted is True
    assert res.rejection_reasons == []


def test_interface_net_unconnected_is_excluded_but_local_still_rejects():
    # An interface (inter-sheet) net routes at parent compose, not in-leaf, so an
    # unconnected SDA is ignored; a *local* signal miss (CC2) still rejects.
    cfg = LeafAcceptanceConfig(max_unconnected=0)
    val = _validation(2, ["SDA", "CC2"])
    val["interface_port_names"] = ["SDA"]
    res = evaluate_leaf_acceptance(val, {}, cfg)
    gate = res.gate_results["no_unconnected"]
    assert gate["signal_unconnected_nets"] == ["CC2"]
    assert gate["ignored_interface_nets"] == ["SDA"]
    assert res.accepted is False  # CC2 (local) still fails


def test_interface_only_unconnected_passes():
    # When every open net is an interface net, the leaf's local routing is done.
    cfg = LeafAcceptanceConfig(max_unconnected=0)
    val = _validation(2, ["USB_D+", "SOIL_ADC"])
    val["interface_port_names"] = ["USB_D+", "SOIL_ADC"]
    res = evaluate_leaf_acceptance(val, {}, cfg)
    gate = res.gate_results["no_unconnected"]
    assert gate["passed"] is True
    assert gate["signal_unconnected_nets"] == []
    assert set(gate["ignored_interface_nets"]) == {"USB_D+", "SOIL_ADC"}
    assert res.accepted is True


def test_none_threshold_skips_gate():
    cfg = LeafAcceptanceConfig(max_unconnected=None)
    res = evaluate_leaf_acceptance(_validation(5, ["CC2", "UART_TX"]), {}, cfg)
    assert res.gate_results["no_unconnected"].get("skipped") is True
    assert "no_unconnected" not in res.rejection_reasons


def test_poured_nets_extends_exclusion():
    # An explicitly-poured signal-named net is treated like power/ground.
    cfg = LeafAcceptanceConfig(max_unconnected=0, poured_nets=frozenset({"CC2"}))
    res = evaluate_leaf_acceptance(_validation(1, ["CC2"]), {}, cfg)
    assert res.accepted is True


def test_fallback_to_raw_count_when_net_names_missing():
    # Format drift: items present but no parsable net names -> surface the miss.
    cfg = LeafAcceptanceConfig(max_unconnected=0)
    res = evaluate_leaf_acceptance(_validation(2, []), {}, cfg)
    assert res.accepted is False


def test_config_from_dict_maps_keys():
    cfg = acceptance_config_from_dict(
        {"leaf_acceptance_max_unconnected": 0, "leaf_acceptance_poured_nets": ["+5V"]}
    )
    assert cfg.max_unconnected == 0
    assert cfg.poured_nets == frozenset({"+5V"})

    cfg_none = acceptance_config_from_dict({"leaf_acceptance_max_unconnected": None})
    assert cfg_none.max_unconnected is None
