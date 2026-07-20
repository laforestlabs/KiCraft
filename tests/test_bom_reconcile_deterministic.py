"""Guards for the deterministic reconcile passive-add (self-eval 2026-07-17
T4, docs/plans/self-eval-2026-07-17-fix-plan.md).

Three briefs in batch 20260717T035619Z died with "unresolved BOM deficit
after 3 reconcile pass(es)" although each deficit note fully specified a
jellybean passive. The executor now provisions parseable asks directly into
``state.bom.parts`` (cloning a same-value donor part, falling back to the
offline catalog) and re-drives wiring only. These tests pin the parser on the
three REAL batch deficit strings and the donor-clone/dedup behavior.
"""
from __future__ import annotations

import json

import kicraft.server.session as session


# The exact design_error texts from batch 20260717T035619Z.
RUN_10 = (
    "The RP2040 (U2) VREG_VOUT (pin 45) requires a 1uF capacitor to GND; the "
    "BOM does not have a dedicated cap for this pin. Add one 1uF 0402/0603 "
    "capacitor on the RP2040 sheet, connected between VREG_VOUT and GND, "
    "clustered with U2."
)
RUN_18 = (
    "The TPS54160 (U1) and U2 each require a feedback voltage divider "
    "resistor from VSENSE (pin7) to GND. The BOM has the top resistors R1 "
    "(140k) and R4 (150k) but is missing the bottom resistors. Add two 10k "
    "resistors (0402/0603) on the DC DC CONVERTER sheet."
)
RUN_22 = (
    "The TPS54331 (U1) needs a 0.1µF capacitor between BOOT (pin1) and "
    "PH (pin8) for bootstrapping, and a 10k resistor and 0.1µF capacitor "
    "in series from COMP (pin6) to GND for compensation. Each DRV8833 (U3, "
    "U4) needs a 0.1µF capacitor between VCP (pin11) and VM (pin12) for "
    "charge pump decoupling. Add these components on the BUCK 3V3 and MOTOR "
    "DRIVER sheets, clustered with the respective ICs."
)


def test_parse_run10_one_1uf_cap_on_named_sheet():
    asks = session.parse_passive_deficits([RUN_10])
    caps = [a for a in asks if a["kind"] == "capacitor"]
    assert caps, "the 1uF ask must parse"
    assert all(session._norm_value(a["value"]) == "1uf" for a in caps)
    assert any(a["sheet"] == "RP2040" for a in caps)


def test_parse_run18_two_10k_resistors():
    asks = session.parse_passive_deficits([RUN_18])
    rs = [a for a in asks if a["kind"] == "resistor"
          and session._norm_value(a["value"]) == "10k"]
    assert rs
    assert max(a["qty"] for a in rs) == 2


def test_parse_run22_mixed_asks_and_ambiguous_sheets():
    asks = session.parse_passive_deficits([RUN_22])
    kinds = {(a["kind"], session._norm_value(a["value"])) for a in asks}
    assert ("capacitor", "0.1uf") in kinds
    assert ("resistor", "10k") in kinds
    # "on the ... and ... sheets" is ambiguous -> no sheet is pinned.
    assert all(a["sheet"] is None for a in asks)


def _ws_with_state(tmp_path, parts, ic_groups=None):
    state = {
        "bom": {
            "parts": parts,
            "ic_groups": ic_groups or {},
            "connections": [],
        }
    }
    kdir = tmp_path / ".kicraft"
    kdir.mkdir()
    (kdir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return tmp_path


def _donor(ref="C1", value="1uF", sheet="POWER"):
    return {
        "ref": ref, "value": value, "symbol": "Device:C",
        "footprint": "Capacitor_SMD:C_0603_1608Metric", "sheet": sheet,
        "mpn": None, "datasheet": None, "sourcing_note": "LCSC C5199872",
        "side": None, "source_leaf": None,
    }


def test_apply_clones_donor_with_fresh_ref_and_target_sheet(tmp_path):
    # U2 lives on the RP2040 sheet, so the ask's scraped sheet name validates
    # (a scraped sheet that matches no existing part sheet is ignored).
    ws = _ws_with_state(
        tmp_path,
        [_donor(),
         {"ref": "U2", "symbol": "MCU:RP2040", "value": "RP2040",
          "footprint": "QFN-56", "sheet": "RP2040", "mpn": "RP2040",
          "datasheet": None, "sourcing_note": None, "side": None,
          "source_leaf": None}],
        ic_groups={"U2": ["C1"]},
    )
    added = session.apply_deterministic_bom_adds(
        ws, [{"text": RUN_10, "reconcile_target": "bom"}]
    )
    assert added == ["C2"]
    state = json.loads((ws / ".kicraft" / "state.json").read_text())
    parts = state["bom"]["parts"]
    assert len(parts) == 3
    new = parts[2]
    assert new["ref"] == "C2"
    assert new["sourcing_note"] == "LCSC C5199872"  # donor's proven sourcing
    assert new["sheet"] == "RP2040"                 # the ask's named sheet
    assert parts[0] == _donor()                     # nothing existing touched


def test_apply_skips_when_prior_add_still_unconsumed(tmp_path):
    # An ungrouped same-value part means an earlier deterministic pass already
    # provisioned this ask and wiring STILL parks -- do not add another copy.
    ws = _ws_with_state(
        tmp_path,
        [_donor(), _donor(ref="C2", sheet="RP2040")],
        ic_groups={"U2": ["C1"]},  # C2 is ungrouped = unconsumed
    )
    added = session.apply_deterministic_bom_adds(
        ws, [{"text": RUN_10, "reconcile_target": "bom"}]
    )
    assert added == []


def test_apply_returns_empty_on_unparseable_text(tmp_path):
    ws = _ws_with_state(tmp_path, [_donor()])
    added = session.apply_deterministic_bom_adds(
        ws, [{"text": "the flux capacitor is sad", "reconcile_target": "bom"}]
    )
    assert added == []


# --- 2026-07-19 review §5.8: partial fulfillment must not read as full ------

RUN_639 = (
    "The design requires a 40MHz crystal (X1) with two 18pF load capacitors "
    "(C8, C9), three additional 100nF decoupling capacitors (C10, C11, C12), "
    "and an antenna connection via a 0-ohm resistor (R7) and a u.FL "
    "connector (J3) on the ESP32 C3 sheet."
)


def test_parse_zero_ohm_resistor():
    asks = session.parse_passive_deficits([RUN_639])
    rs = [a for a in asks if a["kind"] == "resistor"]
    assert rs, "the hyphenated 0-ohm ask must parse"


def test_non_passive_remainder_detected():
    assert session._NON_PASSIVE_ASK_RE.search(RUN_639)
    # crystal AND u.FL AND connector all present; a purely-passive note is clean
    assert not session._NON_PASSIVE_ASK_RE.search(RUN_22)


def test_reconcile_falls_through_on_non_passive_remainder(tmp_path, monkeypatch):
    # added-something must NOT trigger the wiring-only "do NOT park again"
    # path when the note also asks for parts the deterministic pass cannot
    # provision -- the LLM bom+wiring pass owns the remainder.
    calls = []

    def _fake_run_session(ws, brief, stages, **kw):
        calls.append(list(stages))
        return {"ok": True}

    monkeypatch.setattr(session, "run_session", _fake_run_session)
    monkeypatch.setattr(
        session, "apply_deterministic_bom_adds", lambda ws, d: ["C10"]
    )
    monkeypatch.setattr(
        session, "bom_reconcile_deficits",
        lambda res: [{"text": RUN_639}],
    )
    session.maybe_bom_reconcile(tmp_path, "brief", {"ok": False})
    assert calls, "a reconcile pass must run"
    assert calls[0] != ["wiring"], (
        "wiring-only re-drive would falsely claim the deficit was fulfilled"
    )
