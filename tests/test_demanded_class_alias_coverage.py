"""Demanded-class coverage aliases: every alias is proved by a reviewed record.

The 2026-09-29 demanded-class scan of the saved states found the top uncovered classes are the
library's OWN reviewed part under a purpose spelling. An alias is added only when a reviewed
record realizes the demand through `reviewed_record_realizes_class` (family or feature
intersection), never by a spelling heuristic; a class no record carries stays uncovered.
"""

import pytest

from kicraft.design import lowering as lowering_module
from kicraft.design.part_identity import (
    _DEMANDED_CLASS_ALIASES,
    canonical_physical_features,
    has_reviewed_coverage,
    lowerer_witnesses_physical_class,
    realizable_physical_features,
    reviewed_part,
    reviewed_record_realizes_class,
)
from kicraft.design.stage_semantics import complete_class_spellings

# (demanded class, the reviewed identity that proves it, the targets the alias resolves to)
ALIASES = [
    ("qspi-flash", "W25Q16JVSS", {"flash-memory"}),
    ("qspi-flash-memory", "W25Q16JVSS", {"flash-memory"}),
    ("qspi-nor-flash", "W25Q16JVSS", {"flash-memory"}),
    ("servo-driver-ic", "PCA9685PW,118", {"pwm-driver", "i2c-pwm-driver"}),
    ("servo-controller-ic", "PCA9685PW,118", {"pwm-driver", "i2c-pwm-driver"}),
    ("pwm-controller", "PCA9685PW,118", {"pwm-driver", "i2c-pwm-driver"}),
    ("pwm-controller-ic", "PCA9685PW,118", {"pwm-driver", "i2c-pwm-driver"}),
    ("pca9685-controller", "PCA9685PW,118", {"pwm-driver", "i2c-pwm-driver"}),
    ("pca9685-pwm-controller", "PCA9685PW,118", {"pwm-driver", "i2c-pwm-driver"}),
    ("pca9685-ic", "PCA9685PW,118", {"pwm-driver", "i2c-pwm-driver"}),
    ("thermocouple-amplifier", "MAX31855KASA+", {"thermocouple-converter"}),
    ("thermocouple-amplifier-ic", "MAX31855KASA+", {"thermocouple-converter"}),
    ("k-type-thermocouple-input", "MAX31855KASA+", {"thermocouple-converter"}),
    ("optoisolator", "PC817C-S", {"optocoupler"}),
    ("transistor-array", "ULN2003ADR", {"darlington-array"}),
    ("smt-uln2003-driver", "ULN2003ADR", {"darlington-array"}),
    ("smt-uln2003", "ULN2003ADR", {"darlington-array"}),
    ("smt-relay-driver", "ULN2003ADR", {"darlington-array"}),
    ("driver-ic", "ULN2003ADR", {"darlington-array"}),
    ("radio-module", "DL-RFM95-868M", {"sx1276-module", "lora-module", "spi-radio-module"}),
    ("bluetooth-low-energy-radio", "NRF52840-QIAA-R", {"bluetooth-le-soc"}),
    ("ble-radio", "NRF52840-QIAA-R", {"bluetooth-le-soc"}),
    ("motor-connector", "S2B-PH-SM4-TB(LF)(SN)", {"pin-header", "wire-to-board-connector"}),
    ("qwiic-header", "SM04B-SRSS-TB(LF)(SN)", {"qwiic-connector", "i2c-connector"}),
    (
        "inductor",
        "dayton-lw18-50",
        {"air-core-inductor", "power-inductor", "filter-inductor", "buck-inductor",
         "inductor-0402"},
    ),
    (
        "resistor",
        "ERJ-3EKF1002V",
        {"resistor-0603", "resistor-0402", "resistor-1206", "resistor-1210", "resistor-2512"},
    ),
    (
        "capacitor",
        "GRM188R71C104KA01D",
        {"capacitor-0603", "capacitor-0402", "capacitor-0805", "capacitor-1210",
         "film-capacitor", "timing-capacitor", "electrolytic-capacitor",
         "tantalum-capacitor"},
    ),
    ("termination-resistor", "ERJ-3EKF1002V", {"resistor-0603", "resistor-1206"}),
    ("rotary-encoder-push-button", "EC11E15244G1", {"rotary-encoder"}),
    ("three-position-switch", "SS13D07VG4", {"three-position-selector", "sp3t-selector"}),
    (
        "microcontroller-module",
        "ESP32-C3-MINI-1-N4",
        {"esp32-c3-module", "wifi-module", "bluetooth-le-soc"},
    ),
    ("isolated-power-supply", "B0509S-1WR3", {"isolated-dc-dc-converter"}),
    ("switching-regulator-ic", "TLV62569DBVR", {"buck-regulator", "voltage-regulator"}),
    ("buck-regulator-ic", "TLV62569DBVR", {"buck-regulator"}),
    ("smt-i2c-oled", "HS96L03W2C03", {"i2c-oled-display"}),
    ("constant-current-driver", "AL8860MP-13", {"constant-current-led-driver"}),
    ("pd-trigger-controller", "CH224K", {"usb-pd-controller"}),
    ("per-port-current-limiter", "TPS2553DBVR", {"current-limited-power-switch"}),
    ("0805-led", "E6C0805WWAY1UDA(1.1T M)", {"led-0805"}),
    ("warm-white-led-group", "E6C0805WWAY1UDA(1.1T M)", {"warm-white-led"}),
    ("cr2032-holder", "BS-07-A1BJ001", {"coin-cell-holder"}),
    ("cr2032-battery-holder", "BS-07-A1BJ001", {"coin-cell-holder"}),
]


@pytest.mark.parametrize(("demanded", "identity", "targets"), ALIASES)
def test_each_demanded_class_alias_names_the_reviewed_record_that_proves_it(
    demanded, identity, targets
):
    """The alias resolves the demand to the reviewed part's own features, and to them only."""
    record = reviewed_part(identity)
    assert record is not None, f"no reviewed record for {identity}"
    assert canonical_physical_features(demanded) == frozenset(targets), demanded
    assert has_reviewed_coverage(demanded), demanded
    # The library's own record decides, not the spelling: the named part realizes the demand.
    assert reviewed_record_realizes_class(record, demanded), (demanded, identity)


def test_the_demanded_class_lookup_resolves_through_the_class_key_fold():
    """A plural or space spelling reaches the alias key or the reviewed feature it denotes.

    `0805 led` folds to the `0805-led` alias key; `warm-white leds` folds straight onto the
    reviewed `warm-white-led` feature. The caller's raw spelling is kept too, because two
    reviewed features differ from their folded key (`2.4ghz-antenna`, `jst-xh connector`) and
    folding must not uncover a class the library already answers.
    """
    led = reviewed_part("E6C0805WWAY1UDA(1.1T M)")
    assert canonical_physical_features("0805 led") == frozenset({"led-0805"})
    assert canonical_physical_features("0805 LED") == frozenset({"led-0805"})
    assert reviewed_record_realizes_class(led, "0805 led")
    assert canonical_physical_features("warm-white leds") == frozenset(
        {"warm-white leds", "warm-white-led"}
    )
    assert reviewed_record_realizes_class(led, "warm-white leds")
    # `qspi flash` folds onto the alias key and still resolves to the reviewed part.
    assert canonical_physical_features("qspi flash") == frozenset({"flash-memory"})
    assert reviewed_record_realizes_class(reviewed_part("W25Q16JVSS"), "qspi flash")
    # A demand the library already answers under its exact spelling is never uncovered.
    assert "2.4ghz-antenna" in canonical_physical_features("2.4ghz-antenna")
    assert has_reviewed_coverage("2.4ghz-antenna")
    assert "jst-xh connector" in canonical_physical_features("jst-xh connector")
    assert has_reviewed_coverage("jst-xh connector")


@pytest.mark.parametrize(
    ("demanded", "target"),
    [
        # Each of these would otherwise be repaired by `complete_class_spellings` to a
        # strict-token-subset reviewed class that is a DIFFERENT physical part.
        ("smt-relay-driver", "relay"),
        ("rotary-encoder-push-button", "push-button"),
        ("qwiic-header", "header"),
        ("microcontroller-module", "microcontroller"),
        ("three-position-switch", "switch"),
        ("motor-connector", "connector"),
    ],
)
def test_a_hazardous_variant_rename_is_pre_empted_by_the_alias(demanded, target):
    """The alias must fire before the token-subset rename picks the wrong class.

    A relay is not its driver, a momentary button is not a rotary encoder, a 0.1 in header is
    not a Qwiic JST-SH, a bare ATtiny is not an MCU module, and a momentary `switch` is not a
    latching 3-position selector -- so the demand keeps its own spelling instead of being read
    as the wrong reviewed class (and no "defaulted" assumption is recorded).
    """
    # The wrong target is exactly the realizable strict-token-subset variant the rename would
    # have picked had the alias not made the demand realizable first.
    assert target in _renamed_targets_without_the_alias(demanded)
    completed = complete_class_spellings(
        {"obligations": [{"kind": "physical", "component_class": demanded}]}
    )
    assert completed["obligations"][0]["component_class"] == demanded
    assert not completed.get("assumptions")


def _renamed_targets_without_the_alias(demanded: str) -> set[str]:
    """The strict-token-subset reviewed class the rename would choose if the alias were absent."""
    from kicraft.design.part_identity import reviewed_class_variants

    return {
        name for name in reviewed_class_variants(demanded) if realizable_physical_features(name)
    }


def test_a_class_no_reviewed_record_carries_stays_uncovered():
    """Aliasing never invents a carrier: a role spellings with no reviewed part stays a gap.

    `jumper` is left alone deliberately: the corpus uses it both for a removable shunt on a
    0.1 in header (the RS-485 DE/RE jumper) and for a solder jumper (the CAN-termination
    reference's `SolderJumper_2_Open`), which is NOT a pin header -- so it does not denote one
    reviewed physical class. `ground-connector` is a wiring fact, not a part.
    """
    for demanded in ("jumper", "ground-connector", "solder-jumper"):
        assert not has_reviewed_coverage(demanded), demanded
        assert not realizable_physical_features(demanded), demanded
        assert demanded not in _DEMANDED_CLASS_ALIASES, demanded


@pytest.mark.parametrize("spelling", ["r-2r-resistor-ladder", "r2r-resistor-ladder"])
def test_the_r2r_ladder_witness_and_family_agree_on_the_hyphenated_spelling(spelling):
    """The lowerer both answers the demand and proves the class under the same spelling.

    `lowerer_witnesses_physical_class` and `lowering._FAMILIES` key on the exact (resp.
    normalised) class string, so an alias cannot repair this: both spellings must be registered.
    """
    assert lowerer_witnesses_physical_class("r2r-ladder@1", spelling)
    assert lowering_module._FAMILIES.get(spelling) == "r2r-ladder@1"
