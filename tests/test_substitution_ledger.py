"""§9.33 spec-named part accountability + §9.34 brief-stated mount type +
the mcu_programming_facts digest helper (2026-07-27 fix-plan P2,
docs/plans/self-eval-2026-07-27-fix-plan.md).

Batch 20260727T045000Z capped 6 runs on silent_substitution: the BOM walked
away from spec-named parts (RECOM RP12-2412DA -> 125 mA WRA2412S-3WR2) or a
brief-stated attribute (SMT OLED shipped through-hole) with nothing on the
record. These checks make the silence, not the swap, the rejected thing.
Fixture texts are verbatim from that batch where noted.
"""
from __future__ import annotations

from types import SimpleNamespace

from kicraft.design.models import BOM, BomPart, IntentSlot, Substitution
from kicraft.design.synthesis.validation import (
    _spec_named_tokens,
    check_mount_type_consistency,
    check_spec_named_mpn_substitutions,
    mcu_programming_facts,
)


def _arch(assumptions=(), sheets=(), topologies=None):
    return SimpleNamespace(
        assumptions=list(assumptions),
        topologies=topologies or {},
        sheets=[SimpleNamespace(function=s) for s in sheets],
    )


def _fs(assumptions=(), purposes=()):
    return SimpleNamespace(
        assumptions=list(assumptions),
        blocks=[SimpleNamespace(purpose=p) for p in purposes],
    )


def _part(ref="U1", value="X", mpn=None, note=None,
          footprint="Resistor_SMD:R_0603_1608Metric", symbol="Device:R"):
    return BomPart(ref=ref, value=value, symbol=symbol, footprint=footprint,
                   sheet="MAIN", mpn=mpn, sourcing_note=note)


def _bom(parts, substitutions=(), assumptions=()):
    return BOM(parts=parts, substitutions=list(substitutions),
               assumptions=list(assumptions))


# --- token extraction ------------------------------------------------------

def test_spec_tokens_find_real_mpns_and_skip_noise():
    arch = _arch(
        assumptions=[
            # run_18 (verbatim shape): the architecture names the converter.
            "Isolated DC-DC: RECOM RP12-2412DA dual-output module (defaulted)",
        ],
        sheets=["STM32F103C8T6 MCU with decoupling on LQFP48, GPIO12 spare, "
                "I2C1 to sensors"],
    )
    toks = _spec_named_tokens(None, arch)
    assert "rp12-2412da" in toks
    assert "stm32f103c8t6" in toks
    # Pin/package/protocol noise never counts as a named part.
    assert "lqfp48" not in toks
    assert "gpio12" not in toks
    assert "i2c1" not in toks


# --- §9.33 -----------------------------------------------------------------

def test_spec_named_mpn_missing_without_ledger_fails():
    # run_18: RP12-2412DA named, WRA2412S-3WR2 shipped, nothing recorded.
    arch = _arch(assumptions=["Use a RECOM RP12-2412DA isolated module (defaulted)"])
    bom = _bom([_part("PS1", "WRA2412S-3WR2", mpn="WRA2412S-3WR2")])
    r = check_spec_named_mpn_substitutions(None, arch, bom)
    assert not r.ok
    assert any("RP12-2412DA" in o for o in r.offenders)


def test_spec_named_mpn_shipped_passes():
    arch = _arch(assumptions=["Use a RECOM RP12-2412DA isolated module (defaulted)"])
    bom = _bom([_part("PS1", "RP12-2412DA", mpn="RP12-2412DA")])
    assert check_spec_named_mpn_substitutions(None, arch, bom).ok


def test_spec_named_mpn_ledgered_passes():
    arch = _arch(assumptions=["Use a RECOM RP12-2412DA isolated module (defaulted)"])
    bom = _bom(
        [_part("PS1", "WRA2412S-3WR2", mpn="WRA2412S-3WR2")],
        substitutions=[Substitution(
            wanted="RP12-2412DA", got="WRA2412S-3WR2",
            reason="named module not orderable; NOTE 125mA/rail vs 500mA")],
    )
    assert check_spec_named_mpn_substitutions(None, arch, bom).ok


def test_spec_named_mpn_in_assumptions_counts_as_surfaced():
    # run_32's shape: the swap note lives in bom.assumptions -- that IS a
    # record; the gate condition is silence.
    arch = _arch(assumptions=["Connector: SM04B-SRSS-TB per spec (defaulted)"])
    bom = _bom(
        [_part("J1", "XY-SM04B-clone")],
        assumptions=["J1 switched to XY clone of SM04B-SRSS-TB after commit "
                     "rejection (defaulted)"],
    )
    assert check_spec_named_mpn_substitutions(None, arch, bom).ok


def test_functional_spec_purpose_tokens_enforced():
    fs = _fs(purposes=["Thermocouple amplifier around a MAX31855 (defaulted)"])
    bom = _bom([_part("U2", "MAX6675", mpn="MAX6675")])
    r = check_spec_named_mpn_substitutions(fs, None, bom)
    assert not r.ok and any("MAX31855" in o for o in r.offenders)


# --- §9.34 -----------------------------------------------------------------

# run_20 (verbatim brief shape): TH encoder asked AND honored; SMT OLED asked
# and contradicted by the shipped through-hole footprint.
_RUN20_GOAL = ("A front-panel board: a through-hole rotary encoder with push "
               "button, an SMT I2C OLED display, and a 10-pin FFC connector.")


def _run20_intent():
    return IntentSlot(goal=_RUN20_GOAL)


def test_mount_type_contradiction_fails_and_names_the_part():
    bom = _bom([
        _part("U2", "HS96L03W2C03 OLED", symbol="Display:OLED",
              footprint="OLED-TH:OLED-TH_L27.8-W27.2-P2.54"),
        _part("SW1", "Rotary Encoder", symbol="Device:Rotary_Encoder_Switch",
              footprint="Rotary_Encoder:RotaryEncoder_Alps_EC11E_Vertical_THT"),
    ])
    r = check_mount_type_consistency(_run20_intent(), bom)
    assert not r.ok
    assert len(r.offenders) == 1 and "U2" in r.offenders[0]
    # The through-hole encoder honors its own qualifier -- never an offender.
    assert not any("SW1" in o for o in r.offenders)


def test_mount_type_contradiction_ledgered_passes():
    bom = _bom(
        [_part("U2", "HS96L03W2C03 OLED", symbol="Display:OLED",
               footprint="OLED-TH:OLED-TH_L27.8-W27.2-P2.54")],
        substitutions=[Substitution(
            wanted="SMT I2C OLED", got="through-hole OLED HS96L03W2C03",
            reason="no SMT variant in stock")],
    )
    assert check_mount_type_consistency(_run20_intent(), bom).ok


def test_mount_type_matching_footprint_passes():
    bom = _bom([_part("U2", "SSD1306 OLED", symbol="Display:OLED",
                      footprint="Display:OLED_SSD1306_SMD")])
    assert check_mount_type_consistency(_run20_intent(), bom).ok


def test_mount_type_generic_noun_never_fires():
    # "surface-mount components" qualifies nothing specific -- must not sweep
    # the whole BOM.
    intent = IntentSlot(goal="Use surface-mount components where possible.")
    bom = _bom([_part("J1", "DC barrel jack", symbol="Connector:Barrel_Jack",
                      footprint="Connector_BarrelJack:BarrelJack_Horizontal_THT")])
    assert check_mount_type_consistency(intent, bom).ok


def test_mount_type_unclassifiable_footprint_skipped():
    intent = IntentSlot(goal="an SMT buzzer")
    bom = _bom([_part("BZ1", "Buzzer", symbol="Device:Buzzer",
                      footprint="Buzzer:Buzzer_Generic")])
    assert check_mount_type_consistency(intent, bom).ok


# --- mcu_programming_facts (digest surfacing, P2.5) ------------------------

def test_facts_none_without_mcu():
    assert mcu_programming_facts(_bom([_part("R1", "10k")])) is None


def test_facts_report_access_parts_and_verdict():
    bom = _bom([
        _part("U1", "RP2040", symbol="MCU:RP2040",
              footprint="Package_DFN_QFN:QFN-56"),
        _part("J1", "USB-C receptacle", symbol="Connector:USB_C",
              footprint="Connector_USB:USB_C_SMD"),
        _part("SW1", "BOOTSEL button", symbol="Switch:SW_Push",
              footprint="Button_Switch_SMD:SW_SPST"),
    ])
    facts = mcu_programming_facts(bom)
    assert facts is not None and facts["access_ok"]
    assert any("J1" in a for a in facts["access_parts"])
    assert any("U1" in m for m in facts["mcus"])
