"""The congestion-growth valve must fire only for *parent-interconnect*
unconnected nets -- never for leaf-internal ones.

``_parent_rejected_unconnected`` feeds the round scheduler's seed-overhead
growth valve. Growing the parent seed area can only relieve a net that must
route *between* leaves; a net whose endpoints all sit inside one leaf is a
leaf-internal routing failure that no amount of parent area can fix. Letting
those trip the valve is how a dense-SoC leaf (e.g. an nRF52840 subsystem with 6
unroutable decoupling/clock nets) bloats the board to 92% empty and picks up a
copper-edge ``illegal_routed_geometry`` while the leaf nets stay open. See the
KC-69TGAP investigation.
"""

from __future__ import annotations

import json
from pathlib import Path

from kicraft.cli.autoexperiment import _parent_rejected_unconnected


def _write_parent_output(
    tmp_path: Path,
    *,
    unconnected: int,
    unconnected_nets: list[str] | None,
    interconnect_net_names: list[str] | None,
    accepted: bool = False,
) -> Path:
    state: dict = {
        "routed_validation": {
            "accepted": accepted,
            "drc": {"unconnected": unconnected, "unconnected_nets": unconnected_nets},
        },
    }
    if interconnect_net_names is not None:
        state["interconnect_net_names"] = interconnect_net_names
    p = tmp_path / "parent_pipeline.json"
    p.write_text(json.dumps({"state": state}))
    return p


def test_leaf_internal_unconnected_does_not_trip_valve(tmp_path: Path) -> None:
    # The 6 unconnected nets are all inside one leaf; the 3 interconnect nets
    # (GND/VDD/ANT_OUT) are fully routed. Parent area cannot help -> no grow.
    p = _write_parent_output(
        tmp_path,
        unconnected=6,
        unconnected_nets=["DEC1", "DEC3", "DECUSB", "X2_OSC1", "RESET", "BUTTON"],
        interconnect_net_names=["ANT_OUT", "GND", "VDD"],
    )
    assert _parent_rejected_unconnected(p) is False


def test_interconnect_unconnected_trips_valve(tmp_path: Path) -> None:
    # An interconnect net (VDD) is among the unconnected -> a cramped parent
    # placement the seed-overhead growth valve can relieve.
    p = _write_parent_output(
        tmp_path,
        unconnected=2,
        unconnected_nets=["VDD", "DEC1"],
        interconnect_net_names=["ANT_OUT", "GND", "VDD"],
    )
    assert _parent_rejected_unconnected(p) is True


def test_missing_interconnect_names_falls_back_to_any_unconnected(tmp_path: Path) -> None:
    # Older artifact without the membership split: keep the historical
    # any-unconnected-grows behavior rather than silently disabling the valve.
    p = _write_parent_output(
        tmp_path,
        unconnected=3,
        unconnected_nets=["DEC1", "DEC3", "RESET"],
        interconnect_net_names=None,
    )
    assert _parent_rejected_unconnected(p) is True


def test_empty_interconnect_names_falls_back_to_any_unconnected(tmp_path: Path) -> None:
    # A flat / single-leaf board has no cross-leaf net set to classify against.
    # We can't prove the unconnected nets are leaf-internal, so stay
    # conservative and keep the historical grow-on-any behavior (no regression
    # on boards that legitimately benefit from more parent area).
    p = _write_parent_output(
        tmp_path,
        unconnected=2,
        unconnected_nets=["NET1", "NET2"],
        interconnect_net_names=[],
    )
    assert _parent_rejected_unconnected(p) is True


def test_no_net_names_but_count_positive_falls_back(tmp_path: Path) -> None:
    # unconnected count > 0 but the DRC gave no net-name list: can't classify,
    # so keep the historical grow-on-any behavior.
    p = _write_parent_output(
        tmp_path,
        unconnected=4,
        unconnected_nets=None,
        interconnect_net_names=["GND", "VDD"],
    )
    assert _parent_rejected_unconnected(p) is True


def test_accepted_board_never_trips_valve(tmp_path: Path) -> None:
    p = _write_parent_output(
        tmp_path,
        unconnected=0,
        unconnected_nets=[],
        interconnect_net_names=["GND", "VDD"],
        accepted=True,
    )
    assert _parent_rejected_unconnected(p) is False
