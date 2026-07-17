"""Guards for the self-eval 2026-07-17 T8 lints (fix-plan T8):

§9.31 repeated-block coverage — N identical connectors declared, all must be
wired (run_28 shipped 3 of 4 audio jacks electrically inert, every pin NC).

§9.32 regulator feedback divider — deterministic Vout from a known Vref and
the wired divider. Pinned on run_15's REAL topology (TPS5430, 16.9k/10k,
3V3 rail): the judge model hallucinated Vref=0.8V and failed this correct
design; the computed 3.284V fact is the antidote.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from kicraft.design.synthesis.validation import (
    _net_voltage,
    _resistance_ohms,
    check_regulator_feedback_vout,
    check_repeated_block_coverage,
    regulator_vout_facts,
)


def _bom(parts, connections):
    return SimpleNamespace(
        parts=[SimpleNamespace(**p) for p in parts],
        connections=[
            SimpleNamespace(
                net_name=c["net_name"],
                endpoints=[SimpleNamespace(**e) for e in c["endpoints"]],
            )
            for c in connections
        ],
    )


def _jack(ref):
    return {"ref": ref, "symbol": "Connector:AudioJack3", "value": "PJ-320A",
            "footprint": "Connector_Audio:Jack_3.5mm", "mpn": None}


def test_9_31_flags_inert_siblings_of_a_wired_connector():
    parts = [_jack("J2"), _jack("J3"), _jack("J4"), _jack("J5")]
    conns = [
        {"net_name": "CH1", "endpoints": [{"ref": "J2", "pin": "1"}]},
        {"net_name": "GND", "endpoints": [{"ref": "J2", "pin": "2"}]},
    ]
    res = check_repeated_block_coverage(_bom(parts, conns))
    assert not res.ok
    flagged = " ".join(res.offenders)
    assert "J3" in flagged and "J4" in flagged and "J5" in flagged
    assert "J2" not in [o.split()[0] for o in res.offenders]


def test_9_31_ok_when_all_wired_and_ignores_singletons_and_ics():
    parts = [_jack("J2"), _jack("J3"),
             {"ref": "U1", "symbol": "Amplifier:MCP6002", "value": "MCP6002",
              "footprint": "SOIC-8", "mpn": None},
             {"ref": "U2", "symbol": "Amplifier:MCP6002", "value": "MCP6002",
              "footprint": "SOIC-8", "mpn": None}]
    conns = [
        {"net_name": "CH1", "endpoints": [{"ref": "J2", "pin": "1"},
                                          {"ref": "U1", "pin": "3"}]},
        {"net_name": "GND", "endpoints": [{"ref": "J2", "pin": "2"}]},
        {"net_name": "CH2", "endpoints": [{"ref": "J3", "pin": "1"}]},
        {"net_name": "GND2", "endpoints": [{"ref": "J3", "pin": "2"}]},
    ]
    # U2 (an IC spare) is unwired but ICs are excluded by design.
    res = check_repeated_block_coverage(_bom(parts, conns))
    assert res.ok, res.offenders


# run_15's real divider topology, minimally reproduced.
RUN15_PARTS = [
    {"ref": "U1", "symbol": "Regulator:TPS5430", "value": "TPS5430DDAR",
     "footprint": "SOIC-8", "mpn": "TPS5430DDAR"},
    {"ref": "R1", "symbol": "Device:R", "value": "16.9k", "footprint": "0603",
     "mpn": None},
    {"ref": "R2", "symbol": "Device:R", "value": "10k", "footprint": "0603",
     "mpn": None},
]
RUN15_CONNS = [
    {"net_name": "FB", "endpoints": [{"ref": "U1", "pin": "4"},
                                     {"ref": "R1", "pin": "2"},
                                     {"ref": "R2", "pin": "1"}]},
    {"net_name": "3V3", "endpoints": [{"ref": "R1", "pin": "1"}]},
    {"net_name": "GND", "endpoints": [{"ref": "R2", "pin": "2"}]},
]


def test_9_32_computes_the_true_tps5430_vout_and_passes():
    facts = regulator_vout_facts(RUN15_PARTS, RUN15_CONNS)
    assert len(facts) == 1
    f = facts[0]
    assert f["vref"] == 1.221
    assert f["vout"] == pytest.approx(3.284, abs=0.001)
    assert f["ok"] is True
    res = check_regulator_feedback_vout(_bom(RUN15_PARTS, RUN15_CONNS))
    assert res.ok


def test_9_32_flags_a_divider_that_misses_its_named_rail():
    # 10k/10k on a TPS5430 -> 2.442V against a 3V3 net: a real wrong-rail bug.
    parts = [dict(p) for p in RUN15_PARTS]
    parts[1] = dict(parts[1], value="10k")
    res = check_regulator_feedback_vout(_bom(parts, RUN15_CONNS))
    assert not res.ok
    assert "2.442" in res.offenders[0]
    assert "R_top" in res.offenders[0]  # suggests the correcting value


def test_9_32_never_guesses_on_ambiguity_or_unknown_parts():
    # Unknown regulator MPN -> no fact, check passes.
    parts = [dict(RUN15_PARTS[0], mpn="LTC9999"), RUN15_PARTS[1], RUN15_PARTS[2]]
    assert regulator_vout_facts(parts, RUN15_CONNS) == []
    # Missing bottom resistor (run_18's actual deficit) -> no divider, no fact.
    conns_no_bot = [RUN15_CONNS[0], RUN15_CONNS[1]]
    assert regulator_vout_facts(RUN15_PARTS, conns_no_bot) == []


def test_value_and_net_parsers():
    assert _resistance_ohms("16.9k") == pytest.approx(16900)
    assert _resistance_ohms("4k7") == pytest.approx(4700)
    assert _resistance_ohms("1M") == pytest.approx(1e6)
    assert _resistance_ohms("470") == pytest.approx(470)
    # Lowercase m is MILLI (a 100m current-sense shunt), not mega: reading it
    # as 10^8 ohms once picked wrong dividers (review finding).
    assert _resistance_ohms("100m") == pytest.approx(0.1)
    assert _resistance_ohms("not-a-value") is None
    assert _net_voltage("3V3") == pytest.approx(3.3)
    assert _net_voltage("1V25") == pytest.approx(1.25)
    assert _net_voltage("+5V") == pytest.approx(5.0)
    assert _net_voltage("VOUT_12V") == pytest.approx(12.0)
    # '1.25V' once parsed as 25.0 -- the false-MISMATCH class this check was
    # written to eliminate (review finding).
    assert _net_voltage("VOUT_1.25V") == pytest.approx(1.25)
    assert _net_voltage("3.3V_RAIL") == pytest.approx(3.3)
    assert _net_voltage("SDA") is None
