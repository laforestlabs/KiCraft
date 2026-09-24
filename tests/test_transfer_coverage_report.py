"""Tests for the reviewed power-transfer coverage report.

The report is the standing check for the §9.39 hole class: a reviewed record that
*looks* covered but can never produce a graph edge (an ambiguous pin name, an
identity-only record), a power family with no transfer fact at all, and a recipe
that emits a converter whose reviewed evidence is absent. Its classification is
the deliverable, so it is asserted directly rather than by shape.
"""

from __future__ import annotations

from kicraft.cli.transfer_coverage_report import format_report, summarize


def _by_identity(summary: dict) -> dict:
    return {row["identity"]: row for row in summary["records"]}


def test_report_classifies_records_and_recipe_parts():
    summary = summarize()
    rows = _by_identity(summary)

    # The record that unblocked the frozen 18 V -> 3V3 wiring state: its datasheet
    # transfer resolves VIN(3) -> SW(5) against its own vendored symbol.
    ap63203 = rows["ap63203wu-7"]
    assert ap63203["status"] == "edge-producible"
    assert [(p["from_pin"], p["to_pin"]) for p in ap63203["paths"]] == [("3", "5")]

    # The isolated module's path is the module-level one, and it is isolated.
    assert rows["b0509s-1wr3"]["status"] == "edge-producible"
    assert rows["b0509s-1wr3"]["transfer"]["isolated"] is True

    # A transfer whose pin names match two pins of the symbol can never make an edge.
    assert rows["al8860mp-13"]["status"] == "dead"
    assert "resolves both ends" in rows["al8860mp-13"]["reason"]
    # An identity-only record is not a portable candidate, so its fact is unreachable.
    assert rows["tps5430"]["status"] == "dead"
    assert "identity-only" in rows["tps5430"]["reason"]
    # A power path with no fact at all.
    assert rows["ch224k"]["status"] == "no-fact"
    assert {row["identity"] for row in summary["dead"]} >= {"al8860mp-13", "tps5430", "tps5430dda"}
    assert {row["identity"] for row in summary["no_fact"]} >= {"ch224k", "tps2553dbvr"}

    # Recipe-emitted converters, seen from the other end: AP63203's recipe part now has
    # a transfer, while the LDO chargers still have no reviewed record at all.
    recipe = {(row["recipe"], row["part"]): row for row in summary["recipes"]}
    assert recipe[("ap63203-3v3@1", "AP63203WU-7")]["transfer"] is True
    assert recipe[("ap63205-5v@1", "AP63205WU-7")]["transfer"] is True
    assert recipe[("ams1117-3v3@1", "AMS1117-3.3")]["reviewed"] is False
    assert {row["part"] for row in summary["unwitnessed_recipe_parts"]} >= {
        "CH224K",
        "TP4056",
        "MCP1700T-3302E/TT",
        "AMS1117-3.3",
    }


def test_report_renders_every_section():
    text = format_report(summarize())
    for heading in (
        "REVIEWED RECORDS",
        "DEAD FACTS",
        "NO FACT",
        "RECIPE POWER PARTS WITHOUT A REVIEWED TRANSFER",
    ):
        assert heading in text
    assert "ap63203wu-7" in text
