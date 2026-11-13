"""CH224K source range is not the internal VDD shunt pin voltage."""
from types import SimpleNamespace

import pytest

from kicraft.design.models import RecipeSelection
from kicraft.design.recipe_versions import legacy_recipe_replacement
from kicraft.design.recipes.registry import expand_recipe
from kicraft.design.synthesis.validation import check_reviewed_input_operating_ranges


def _expanded(recipe):
    ports = {"vbus": "VBUS", "gnd": "GND"}
    if "selectable" in recipe:
        ports.update(cc1="CC1", cc2="CC2")
    return expand_recipe(RecipeSelection(
        recipe=recipe, instance="pd", sheets={"power": "PD"}, port_bindings=ports,
    ))


@pytest.mark.parametrize("recipe", ["ch224k-pd-trigger@2", "ch224k-pd-selectable@1"])
@pytest.mark.parametrize("defect", [None, "overvoltage", "undervoltage", "untyped", "missing_feed",
                                    "undersized_feed", "wrong_feed", "missing_bypass", "unrated_bypass",
                                    "wrong_sense", "extra_vdd_load", "selected_undersized",
                                    "wrong_footprint", "unreviewed_feed", "wrong_selected_lcsc"])
def test_expanded_shunt_supply_is_proved_from_actual_connections(recipe, defect):
    expansion = _expanded(recipe)
    parts = list(expansion.parts)
    connections = list(expansion.connections)
    roles = {part.recipe_role: part for part in parts}
    feed, bypass, sense = (roles[role] for role in ("vdd_feed", "decoupling", "vbus_sense"))
    rails = {"VBUS": 23.0 if defect == "overvoltage" else 9.0, "GND": 0.0}
    if defect == "undervoltage":
        rails["VBUS"] = 3.9
    if defect == "untyped":
        rails.pop("VBUS")
    if defect in {"missing_feed", "missing_bypass"}:
        removed = feed if defect == "missing_feed" else bypass
        parts = [part for part in parts if part.ref != removed.ref]
        connections = [row.model_copy(update={"endpoints": [ep for ep in row.endpoints if ep.ref != removed.ref]})
                       for row in connections]
    if defect in {"undersized_feed", "wrong_feed", "wrong_sense", "unrated_bypass"}:
        target = sense if defect == "wrong_sense" else bypass if defect == "unrated_bypass" else feed
        value = {"undersized_feed": "1k 1% 0.125W", "wrong_feed": "2k 1% 0.5W",
                 "wrong_sense": "1k", "unrated_bypass": "1uF"}[defect]
        parts = [part.model_copy(update={"value": value}) if part.ref == target.ref else part for part in parts]
    if defect in {"selected_undersized", "wrong_footprint", "unreviewed_feed"}:
        changes = {
            "selected_undersized": {
                "mpn": "AC0603FR-071KL",
                "footprint": "Resistor_SMD:R_0603_1608Metric",
            },
            "wrong_footprint": {"footprint": "Resistor_SMD:R_0603_1608Metric"},
            "unreviewed_feed": {"mpn": "UNREVIEWED-1K"},
        }[defect]
        # Keep the claimed 0.5 W value: the selected physical part must prove it.
        parts = [part.model_copy(update=changes) if part.ref == feed.ref else part for part in parts]
    if defect == "wrong_selected_lcsc":
        parts = [
            part.model_copy(update={"sourcing_note": "LCSC C335010"})
            if part.ref == feed.ref else part for part in parts
        ]
    if defect == "extra_vdd_load":
        vdd = next(row.net_name for row in connections for ep in row.endpoints
                   if ep.ref == roles["controller"].ref and ep.pin == "1")
        connections.append(SimpleNamespace(net_name=vdd, endpoints=[SimpleNamespace(ref="U99", pin="1")]))
    bom = SimpleNamespace(parts=parts, connections=connections)
    result = check_reviewed_input_operating_ranges(SimpleNamespace(rail_voltages=rails), bom)
    assert result.ok is (defect is None), result.offenders
    if defect is not None:
        assert any("E_SHUNT_SUPPLY" in item for item in result.offenders)


def test_fixed_trigger_is_nine_volt_pd_only_and_old_graph_requires_regeneration():
    expansion = _expanded("ch224k-pd-trigger@2")
    parts = {part.recipe_role: part for part in expansion.parts}
    nets = {(ep.ref, ep.pin): row.net_name for row in expansion.connections for ep in row.endpoints}
    controller = parts["controller"].ref
    assert nets[controller, "4"] == nets[controller, "5"]
    assert nets[controller, "9"] == nets[parts["config"].ref, "1"]
    assert nets[parts["config"].ref, "2"] == "GND"
    assert parts["config"].value == "6.8k 1%"
    nc = {(pin.ref, pin.pin) for pin in expansion.no_connect_pins}
    assert {(controller, "2"), (controller, "3")} <= nc
    assert all(nets[parts["connector"].ref, str(pin)] == "GND" for pin in range(1, 5))
    assert legacy_recipe_replacement("ch224k-pd-trigger@1") == "ch224k-pd-trigger@2"
