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


# --------------------------------------------------------------------------- #
# 2026-07-27 batch fix-plan P1 (docs/plans/self-eval-2026-07-27-fix-plan.md):
# kind-then-value asks, replacement instructions, per-deficit stuck detection.
# Deficit texts are VERBATIM from batch 20260727T045000Z.
# --------------------------------------------------------------------------- #

RUN_22_FB = (
    "The TPS54331 (U2) on BUCK 3V3 requires a feedback voltage divider from "
    "+3V3 to VSENSE (pin 5). The BOM has only a top resistor R3 (12k). Add a "
    "bottom resistor (e.g., 3.9k, typical for 3.3V output) on the BUCK 3V3 "
    "sheet, connected between VSENSE and GND."
)
RUN_24_BOOT0 = (
    "The STM32F042 (U2) needs a 10k resistor (R6) between BOOT0 (pin 1) and "
    "GND, and a test point (TP1, a single pad) connected to BOOT0 for "
    "entering DFU mode. Add these to the MCU sheet, clustered with U2."
)
RUN_24_TERMINALS = (
    "The design needs 8 analog input channels (AIN0_ADC1 through AIN3_ADC1 "
    "and AIN0_ADC2 through AIN3_ADC2), but the BOM has only 4 x 2-pin screw "
    "terminals (J2-J5) on the INPUT TERMINAL sheet. Add 4 more 2-pin screw "
    "terminals."
)
RUN_08_REPLACE = (
    "Replace U1 (ADuM1301ARWZ) with two ADuM1201ARZ (U1A and U1B) on the "
    "DIGITAL ISOLATOR sheet. Add two 100nF decoupling capacitors (C8 and C9) "
    "on the same sheet for the second isolator's VDD1 and VDD2 supplies."
)


def test_parse_kind_then_value_fb_resistor():
    # run_22 died with this ask falling through to an LLM pass that never
    # applied it: the value comes AFTER the noun, inside an "(e.g., ...)".
    asks = session.parse_passive_deficits([RUN_22_FB])
    rs = [a for a in asks if a["kind"] == "resistor"]
    assert [session._norm_value(a["value"]) for a in rs] == ["3.9k"]
    assert rs[0]["qty"] == 1
    assert rs[0]["sheet"] == "BUCK 3V3"


def test_parse_kind_then_value_never_reads_descriptive_prose():
    # "The BOM has only a top resistor R3 (12k)" carries no ask-verb; the
    # supplement must not provision a duplicate 12k from it.
    asks = session.parse_passive_deficits([RUN_22_FB])
    assert all(session._norm_value(a["value"]) != "12k" for a in asks)


def test_parse_kind_then_value_range_takes_first_value():
    text = ("Add a bottom resistor (e.g., 3.3k to 4.7k, typical for 3.3V "
            "output) on the BUCK 3V3 sheet.")
    asks = session.parse_passive_deficits([text])
    rs = [a for a in asks if a["kind"] == "resistor"]
    assert [session._norm_value(a["value"]) for a in rs] == ["3.3k"]


def test_parse_boot0_sheet_from_to_the_form():
    # "Add these to the MCU sheet" -- the sheet hint uses "to", not "on".
    asks = session.parse_passive_deficits([RUN_24_BOOT0])
    rs = [a for a in asks if a["kind"] == "resistor"
          and session._norm_value(a["value"]) == "10k"]
    assert rs and rs[0]["sheet"] == "MCU"


def test_test_point_and_screw_terminal_are_non_passive_remainders():
    # run_24's BOOT0 note also asks for a TP the deterministic pass can never
    # add; partial fulfillment must fall through to the LLM pass (§5.8).
    assert session._NON_PASSIVE_ASK_RE.search(RUN_24_BOOT0)
    assert session._NON_PASSIVE_ASK_RE.search(RUN_24_TERMINALS)
    # ...but prose like "the positive terminal of C3" must NOT match.
    assert not session._NON_PASSIVE_ASK_RE.search(
        "connect the positive terminal of C3 to GND")


def test_replacement_ask_gets_replacement_instruction():
    ins = session.bom_reconcile_instruction([{"text": RUN_08_REPLACE}])
    assert "remove ONLY the named part(s)" in ins
    assert "do NOT drop any part already present" not in ins
    assert RUN_08_REPLACE.split(".")[0] in ins


def test_add_only_ask_keeps_add_only_instruction():
    ins = session.bom_reconcile_instruction([{"text": RUN_24_BOOT0}])
    assert "do NOT drop any part already present" in ins


def test_deficit_key_stable_across_rewordings_distinct_across_deficits():
    reworded = (
        "The TPS54331 (U2) on BUCK 3V3 requires a feedback voltage divider "
        "from +3V3 to VSENSE (pin 5). The BOM has only a top resistor R3 "
        "(12k). Add a bottom resistor (e.g., 3.3k to 4.7k) on the BUCK 3V3 "
        "sheet, connected between VSENSE and GND."
    )
    k_orig = session._deficit_key([{"text": RUN_22_FB}])
    k_rew = session._deficit_key([{"text": reworded}])
    k_boot = session._deficit_key([{"text": RUN_24_BOOT0}])
    assert k_orig == k_rew, "a reworded repeat is the SAME deficit"
    assert k_orig != k_boot, "a genuinely new deficit must differ"


def _reconcile_harness(tmp_path, monkeypatch, *, mutate_bom, next_deficits):
    """Drive maybe_bom_reconcile once with a scripted LLM pass. ``mutate_bom``
    controls whether the fake pass changes the committed BOM; ``next_deficits``
    is what wiring re-parks on afterwards. Returns (result, passes, calls)."""
    ws = _ws_with_state(tmp_path, [_donor()])
    calls = []

    def _fake_run_session(w, brief, stages, instruction=None, **kw):
        calls.append({"stages": list(stages), "instruction": instruction or ""})
        if mutate_bom:
            state = json.loads((ws / ".kicraft" / "state.json").read_text())
            state["bom"]["parts"].append(
                _donor(ref=f"C{90 + len(calls)}", value="47pF"))
            (ws / ".kicraft" / "state.json").write_text(json.dumps(state))
        return {"status": "awaiting_input", "last_stage": "wiring",
                "questions": next_deficits}

    monkeypatch.setattr(session, "run_session", _fake_run_session)
    monkeypatch.setattr(session, "apply_deterministic_bom_adds",
                        lambda w, d: [])
    res = {"status": "awaiting_input", "last_stage": "wiring",
           "questions": [{"text": RUN_24_TERMINALS,
                          "reconcile_target": "bom"}]}
    rr, passes = session.maybe_bom_reconcile(ws, "brief", res)
    return rr, passes, calls


def test_changed_nothing_but_new_deficit_keeps_budget(tmp_path, monkeypatch):
    # run_24's death: a no-change pass on the terminals deficit burned the
    # WHOLE budget just as wiring surfaced the (new) BOOT0 deficit.
    _, passes, calls = _reconcile_harness(
        tmp_path, monkeypatch, mutate_bom=False,
        next_deficits=[{"text": RUN_24_BOOT0, "reconcile_target": "bom"}])
    assert len(calls) == 1
    assert passes == 1, "an advancing chain must not exhaust the budget"


def test_changed_nothing_same_deficit_is_stuck(tmp_path, monkeypatch):
    _, passes, calls = _reconcile_harness(
        tmp_path, monkeypatch, mutate_bom=False,
        next_deficits=[{"text": RUN_24_TERMINALS,
                        "reconcile_target": "bom"}])
    assert len(calls) == 1
    assert passes == session.BOM_RECONCILE_MAX_PASSES


def test_changed_but_unresolved_gets_one_pointed_retry(tmp_path, monkeypatch):
    # run_22's death: two "ok" BOM passes changed the BOM without ever adding
    # the asked-for part. The retry must quote the unmet ask.
    _, passes, calls = _reconcile_harness(
        tmp_path, monkeypatch, mutate_bom=True,
        next_deficits=[{"text": RUN_24_TERMINALS,
                        "reconcile_target": "bom"}])
    assert len(calls) == 2, "exactly one pointed retry"
    assert "did NOT resolve" in calls[1]["instruction"]
    assert passes == 2
