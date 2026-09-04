"""Tests for §9 mechanical validation checks.

Each check has a positive (passing) and negative (failing) case using small
hand-rolled fixture directories. The LLUPS project (live, slightly quirky) is
used as a real-world smoke fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.design.models import (
    BOM,
    Architecture,
    BomPart,
    InterSheetNet,
    NetConnection,
    PinEndpoint,
    Sheet,
    SheetPin,
)
from kicraft.design.synthesis.validation import (
    SynthesisValidationError,
    check_autoplacer_is_valid_json,
    check_capacitor_polarity_consistency,
    check_footprints_nonempty,
    check_bom_size,
    check_collection_bounds,
    check_inter_sheet_nets_realized,
    check_named_refs_exist,
    check_no_dangling_signal_nets,
    check_pin_directions,
    check_power_pin_polarity,
    check_rf_feed_isolation,
    check_schematic_version,
    check_sheetfile_refs_resolve,
    check_sheets_have_parts,
    check_bom_parts_reference_architecture_sheets,
    check_two_terminal_self_short,
    run_validations,
)

LLUPS_ROOT = Path("/home/jason/Documents/LLUPS")
LLUPS_AVAILABLE = (LLUPS_ROOT / "LLUPS.kicad_sch").is_file()


# ---------- helpers to build a tiny passing project ----------


def _write_minimal_project(d: Path, stem: str = "TINY") -> None:
    """Two-sheet fixture that passes every §9 check."""
    (d / f"{stem}.kicad_sch").write_text(
        f"(kicad_sch\n"
        f"\t(version 20250114)\n"
        f'\t(generator "eeschema")\n'
        f'\t(uuid "11111111-1111-1111-1111-111111111111")\n'
        f"\t(lib_symbols)\n"
        f"\t(sheet\n"
        f"\t\t(at 30 40) (size 30 15)\n"
        f'\t\t(uuid "22222222-2222-2222-2222-222222222222")\n'
        f'\t\t(property "Sheetname" "REG" (at 30 39 0))\n'
        f'\t\t(property "Sheetfile" "REG.kicad_sch" (at 30 56 0))\n'
        f'\t\t(pin "VBUS" bidirectional (at 60 45 0) (uuid "33333333-3333-3333-3333-333333333333"))\n'
        f"\t)\n"
        f")\n"
    )
    (d / "REG.kicad_sch").write_text(
        "(kicad_sch\n"
        "\t(version 20250114)\n"
        '\t(generator "eeschema")\n'
        '\t(uuid "44444444-4444-4444-4444-444444444444")\n'
        "\t(lib_symbols)\n"
        "\t(symbol\n"
        '\t\t(lib_id "Device:R")\n'
        "\t\t(at 100 80 0)\n"
        '\t\t(uuid "55555555-5555-5555-5555-555555555555")\n'
        '\t\t(property "Reference" "R1" (at 100 70 0))\n'
        '\t\t(property "Value" "10k" (at 100 73 0))\n'
        '\t\t(property "Footprint" "Resistor_SMD:R_0402_1005Metric" (at 100 86 0))\n'
        "\t)\n"
        ")\n"
    )
    (d / f"{stem}_autoplacer.json").write_text(
        json.dumps(
            {
                "project_name": stem,
                "pcb_file": f"{stem}.kicad_pcb",
                "power_nets": ["VBUS", "GND"],
                "ic_groups": {"R1": []},
                "signal_flow_order": ["R1"],
            }
        )
    )


# ---------- §9.1 schematic version ----------


def test_version_passes_on_kicad9(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    r = check_schematic_version(tmp_path)
    assert r.ok, r.message


def test_version_fails_on_kicad8(tmp_path: Path) -> None:
    (tmp_path / "X.kicad_sch").write_text("(kicad_sch (version 20230121)\n)\n")
    r = check_schematic_version(tmp_path)
    assert not r.ok
    assert any("20230121" in o for o in r.offenders)


def test_version_fails_when_missing(tmp_path: Path) -> None:
    (tmp_path / "X.kicad_sch").write_text("(kicad_sch (generator x)\n)\n")
    r = check_schematic_version(tmp_path)
    assert not r.ok


# ---------- §9.2 footprints ----------


def test_footprints_pass(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    r = check_footprints_nonempty(tmp_path)
    assert r.ok, r.message


def test_footprints_fail_on_real_instance(tmp_path: Path) -> None:
    (tmp_path / "X.kicad_sch").write_text(
        "(kicad_sch (version 20250114) (lib_symbols)\n"
        '(symbol (lib_id "Device:R") (at 0 0 0)\n'
        '  (property "Reference" "R1" (at 0 0 0))\n'
        '  (property "Footprint" "" (at 0 0 0))\n'
        ")\n)\n"
    )
    r = check_footprints_nonempty(tmp_path)
    assert not r.ok
    assert any("R1" in o for o in r.offenders)


def test_footprints_ignore_lib_symbols_template(tmp_path: Path) -> None:
    # lib_symbols carries an empty Footprint template; placed instance has a real one.
    (tmp_path / "X.kicad_sch").write_text(
        "(kicad_sch (version 20250114)\n"
        "(lib_symbols\n"
        '  (symbol "Device:R"\n'
        '    (property "Reference" "R" (at 0 0 0))\n'
        '    (property "Footprint" "" (at 0 0 0))\n'
        "  )\n"
        ")\n"
        '(symbol (lib_id "Device:R") (at 0 0 0)\n'
        '  (property "Reference" "R1" (at 0 0 0))\n'
        '  (property "Footprint" "Resistor_SMD:R_0402_1005Metric" (at 0 0 0))\n'
        ")\n)\n"
    )
    r = check_footprints_nonempty(tmp_path)
    assert r.ok, r.offenders


def test_footprints_ignore_power_symbols(tmp_path: Path) -> None:
    # #PWR is KiCad's power-flag pseudo-component; its empty Footprint is correct.
    (tmp_path / "X.kicad_sch").write_text(
        "(kicad_sch (version 20250114) (lib_symbols)\n"
        '(symbol (lib_id "power:GND") (at 0 0 0)\n'
        '  (property "Reference" "#PWR0042" (at 0 0 0))\n'
        '  (property "Footprint" "" (at 0 0 0))\n'
        ")\n)\n"
    )
    r = check_footprints_nonempty(tmp_path)
    assert r.ok, r.offenders


# ---------- §9.3 pin directions ----------


def test_pin_directions_pass(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    r = check_pin_directions(tmp_path)
    assert r.ok, r.message


def test_pin_directions_fail(tmp_path: Path) -> None:
    (tmp_path / "X.kicad_sch").write_text(
        "(kicad_sch (version 20250114) (lib_symbols)\n"
        "(sheet (at 0 0) (size 10 5)\n"
        '  (pin "VBUS" power_in (at 0 0 0) (uuid "x"))\n'
        ")\n)\n"
    )
    r = check_pin_directions(tmp_path)
    assert not r.ok
    assert any("power_in" in o for o in r.offenders)


# ---------- §9.4 sheetfile refs ----------


def test_sheetfile_refs_resolve(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    r = check_sheetfile_refs_resolve(tmp_path)
    assert r.ok, r.message


def test_sheetfile_refs_missing(tmp_path: Path) -> None:
    (tmp_path / "X.kicad_sch").write_text(
        "(kicad_sch (version 20250114) (lib_symbols)\n"
        '(sheet (property "Sheetfile" "GHOST.kicad_sch" (at 0 0 0)))\n)\n'
    )
    r = check_sheetfile_refs_resolve(tmp_path)
    assert not r.ok
    assert any("GHOST" in o for o in r.offenders)


# ---------- §9.5 autoplacer JSON ----------


def test_autoplacer_json_valid(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    r = check_autoplacer_is_valid_json(tmp_path, "TINY")
    assert r.ok


def test_autoplacer_json_missing(tmp_path: Path) -> None:
    r = check_autoplacer_is_valid_json(tmp_path, "NONE")
    assert not r.ok


def test_autoplacer_json_malformed(tmp_path: Path) -> None:
    (tmp_path / "X_autoplacer.json").write_text("{not json")
    r = check_autoplacer_is_valid_json(tmp_path, "X")
    assert not r.ok


# ---------- §9.6 autoplacer refs in schematic ----------


def test_named_refs_pass(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    r = check_named_refs_exist(tmp_path, "TINY")
    assert r.ok, r.offenders


def test_named_refs_fail_when_missing(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    # Add a thermal_ref the schematic doesn't have.
    cfg = json.loads((tmp_path / "TINY_autoplacer.json").read_text())
    cfg["thermal_refs"] = ["U999"]
    (tmp_path / "TINY_autoplacer.json").write_text(json.dumps(cfg))
    r = check_named_refs_exist(tmp_path, "TINY")
    assert not r.ok
    assert "U999" in r.offenders


# ---------- aggregator ----------


def test_run_validations_passes(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    results = run_validations(tmp_path, "TINY")
    assert all(r.ok for r in results)


def test_run_validations_raises_on_failure(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path)
    cfg = json.loads((tmp_path / "TINY_autoplacer.json").read_text())
    cfg["thermal_refs"] = ["U999"]
    (tmp_path / "TINY_autoplacer.json").write_text(json.dumps(cfg))
    with pytest.raises(SynthesisValidationError) as exc:
        run_validations(tmp_path, "TINY")
    assert "U999" in str(exc.value)


# ---------- LLUPS real-world smoke ----------


@pytest.mark.skipif(not LLUPS_AVAILABLE, reason="LLUPS project not present")
def test_llups_version_check_passes() -> None:
    assert check_schematic_version(LLUPS_ROOT).ok


@pytest.mark.skipif(not LLUPS_AVAILABLE, reason="LLUPS project not present")
def test_llups_footprints_passes_with_lib_symbols_and_power_flag_filtering() -> None:
    # Regression: naive grep flags lib_symbols templates and #PWR power flags.
    assert check_footprints_nonempty(LLUPS_ROOT).ok


@pytest.mark.skipif(not LLUPS_AVAILABLE, reason="LLUPS project not present")
def test_llups_pin_directions_passes() -> None:
    assert check_pin_directions(LLUPS_ROOT).ok


@pytest.mark.skipif(not LLUPS_AVAILABLE, reason="LLUPS project not present")
def test_llups_sheetfile_refs_passes() -> None:
    assert check_sheetfile_refs_resolve(LLUPS_ROOT).ok


# ---------- §9.13 sheet population + §9.14 inter-sheet net coverage ----------
#
# Model-data checks over (architecture x bom): no files, no symbol lookups, so
# they run without pcbnew. The regression case mirrors the project-3 wireless
# charger, where signal nets PWM_H/PWM_L/COIL_OUT were left unrealized and the
# COIL DRIVER sheet was empty -- caught only by §9.12 ERC at synthesis time.


def _part(ref: str, sheet: str) -> BomPart:
    return BomPart(
        ref=ref,
        value="x",
        symbol="Device:R",
        footprint="Resistor_SMD:R_0402_1005Metric",
        sheet=sheet,
    )


def _bom_with_sheet_counts(*counts: tuple[str, int]) -> BOM:
    parts = []
    index = 1
    for sheet, count in counts:
        for _ in range(count):
            parts.append(_part(f"R{index}", sheet))
            index += 1
    return BOM(parts=parts)


def test_bom_size_preserves_documented_array_scale() -> None:
    result = check_bom_size(_bom_with_sheet_counts(("ARRAY", 400)))
    assert result.ok
    assert result.name == "9.35 BOM emission bounds"


def test_bom_size_rejects_total_overflow() -> None:
    result = check_bom_size(_bom_with_sheet_counts(("LEFT", 251), ("RIGHT", 250)))
    assert not result.ok
    assert result.offenders == ["parts total (501 items, > 500)"]


def test_bom_size_rejects_per_sheet_overflow() -> None:
    result = check_bom_size(_bom_with_sheet_counts(("ARRAY", 451)))
    assert not result.ok
    assert result.offenders == ["ARRAY (451 items, > 450)"]


def test_collection_bounds_sorts_worst_groups_deterministically() -> None:
    items = ["B"] * 5 + ["A"] * 7
    result = check_collection_bounds(
        "parts",
        items,
        total=20,
        per_group=4,
        group_key=lambda item: item,
    )
    assert result.offenders == [
        "A (7 items, > 4)",
        "B (5 items, > 4)",
    ]


def _two_sheet_design() -> tuple[Architecture, BOM]:
    """CONTROLLER -> DRIVER over one signal net (SIG) and one power net (GND),
    both fully wired. Passes §9.13 and §9.14."""
    arch = Architecture(
        sheets=[
            Sheet(name="CONTROLLER", stem="CONTROLLER", function="control"),
            Sheet(name="DRIVER", stem="DRIVER", function="drive"),
        ],
        power_nets=["GND"],
        inter_sheet_nets=[
            InterSheetNet(
                name="SIG",
                endpoints=[
                    SheetPin(sheet="CONTROLLER", direction="output"),
                    SheetPin(sheet="DRIVER", direction="input"),
                ],
            ),
            InterSheetNet(
                name="GND",
                endpoints=[
                    SheetPin(sheet="CONTROLLER", direction="bidirectional"),
                    SheetPin(sheet="DRIVER", direction="bidirectional"),
                ],
            ),
        ],
    )
    bom = BOM(
        parts=[_part("U1", "CONTROLLER"), _part("U2", "DRIVER")],
        connections=[
            NetConnection(
                net_name="SIG", sheet="CONTROLLER", endpoints=[PinEndpoint(ref="U1", pin="1")]
            ),
            NetConnection(
                net_name="SIG", sheet="DRIVER", endpoints=[PinEndpoint(ref="U2", pin="1")]
            ),
            NetConnection(
                net_name="GND", sheet="CONTROLLER", endpoints=[PinEndpoint(ref="U1", pin="2")]
            ),
            NetConnection(
                net_name="GND", sheet="DRIVER", endpoints=[PinEndpoint(ref="U2", pin="2")]
            ),
        ],
    )
    return arch, bom


def test_inter_sheet_realized_passes_when_wired_both_sides() -> None:
    arch, bom = _two_sheet_design()
    assert check_inter_sheet_nets_realized(arch, bom).ok


def test_inter_sheet_realized_flags_unwired_signal_endpoint() -> None:
    arch, bom = _two_sheet_design()
    # Drop the DRIVER side of SIG: the parent sheet pin now has no leaf label.
    bom.connections = [
        c for c in bom.connections if not (c.net_name == "SIG" and c.sheet == "DRIVER")
    ]
    r = check_inter_sheet_nets_realized(arch, bom)
    assert not r.ok
    assert len(r.offenders) == 1
    assert "SIG" in r.offenders[0] and "DRIVER" in r.offenders[0]


def test_inter_sheet_realized_ignores_power_nets() -> None:
    arch, bom = _two_sheet_design()
    # Drop a power-net endpoint: power crosses via global symbols, not sheet
    # pins, so §9.14 must not flag it (§9.11 owns per-pin power coverage).
    bom.connections = [
        c for c in bom.connections if not (c.net_name == "GND" and c.sheet == "DRIVER")
    ]
    assert check_inter_sheet_nets_realized(arch, bom).ok


def test_sheets_have_parts_passes_when_populated() -> None:
    arch, bom = _two_sheet_design()
    assert check_sheets_have_parts(arch, bom).ok


def test_bom_sheet_references_pass_when_all_parts_use_declared_sheets() -> None:
    arch, bom = _two_sheet_design()
    result = check_bom_parts_reference_architecture_sheets(arch, bom)
    assert result.ok
    assert result.name == "9.13 BOM sheet references"


def test_bom_sheet_references_flags_unknown_sheet_without_replacing_inverse_check() -> None:
    arch, bom = _two_sheet_design()
    bom.parts.append(_part("H1", "MOUNTING HOLES"))

    result = check_bom_parts_reference_architecture_sheets(arch, bom)
    assert not result.ok
    assert result.name == "9.13 BOM sheet references"
    assert result.offenders == ["H1 -> 'MOUNTING HOLES'"]
    assert check_sheets_have_parts(arch, bom).ok

    arch.sheets.append(Sheet(name="EMPTY", stem="EMPTY", function="unused"))
    assert not check_sheets_have_parts(arch, bom).ok
    assert result.offenders == ["H1 -> 'MOUNTING HOLES'"]


def test_sheets_have_parts_flags_empty_sheet() -> None:
    arch, bom = _two_sheet_design()
    arch.sheets.append(Sheet(name="COIL DRIVER", stem="COIL_DRIVER", function="drive"))
    r = check_sheets_have_parts(arch, bom)
    assert not r.ok
    assert len(r.offenders) == 1 and "COIL DRIVER" in r.offenders[0]


def test_sheets_have_parts_exempts_library_backed_sheets() -> None:
    arch, bom = _two_sheet_design()
    # A library-backed sheet has no BOM parts of its own (the leaf installer
    # populates it); §9.8 checks its interface, so §9.13 must not flag it.
    arch.sheets.append(
        Sheet(
            name="REG MODULE",
            stem="REG_MODULE",
            function="reg",
            from_library="buck_3v3@1.0",
            library_instance=1,
        )
    )
    assert check_sheets_have_parts(arch, bom).ok


def test_wireless_charger_regression() -> None:
    """The exact project-3 failure: PWM_H/PWM_L/COIL_OUT unrealized on the
    COIL DRIVER (and QI CONTROLLER) side, plus an empty COIL DRIVER sheet."""
    sheets = ["USB C INPUT", "POWER MANAGEMENT", "QI CONTROLLER", "COIL DRIVER", "TRANSMIT COIL"]
    arch = Architecture(
        sheets=[Sheet(name=n, stem=n.replace(" ", "_"), function="f") for n in sheets],
        power_nets=["VBUS", "+3V3", "GND"],
        inter_sheet_nets=[
            InterSheetNet(
                name="VBUS",
                endpoints=[
                    SheetPin(sheet="USB C INPUT", direction="bidirectional"),
                    SheetPin(sheet="POWER MANAGEMENT", direction="bidirectional"),
                    SheetPin(sheet="COIL DRIVER", direction="bidirectional"),
                ],
            ),
            InterSheetNet(
                name="+3V3",
                endpoints=[
                    SheetPin(sheet="POWER MANAGEMENT", direction="bidirectional"),
                    SheetPin(sheet="QI CONTROLLER", direction="bidirectional"),
                ],
            ),
            InterSheetNet(
                name="PWM_H",
                endpoints=[
                    SheetPin(sheet="QI CONTROLLER", direction="output"),
                    SheetPin(sheet="COIL DRIVER", direction="input"),
                ],
            ),
            InterSheetNet(
                name="PWM_L",
                endpoints=[
                    SheetPin(sheet="QI CONTROLLER", direction="output"),
                    SheetPin(sheet="COIL DRIVER", direction="input"),
                ],
            ),
            InterSheetNet(
                name="COIL_OUT",
                endpoints=[
                    SheetPin(sheet="COIL DRIVER", direction="output"),
                    SheetPin(sheet="TRANSMIT COIL", direction="input"),
                ],
            ),
            InterSheetNet(
                name="GND", endpoints=[SheetPin(sheet=s, direction="bidirectional") for s in sheets]
            ),
        ],
    )
    bom = BOM(
        # Every sheet populated EXCEPT COIL DRIVER (the empty-sheet bug).
        parts=[
            _part("U1", "USB C INPUT"),
            _part("U2", "POWER MANAGEMENT"),
            _part("U3", "QI CONTROLLER"),
            _part("L1", "TRANSMIT COIL"),
        ],
        connections=[
            # Only the TRANSMIT COIL side of COIL_OUT ever got wired.
            NetConnection(
                net_name="COIL_OUT",
                sheet="TRANSMIT COIL",
                endpoints=[PinEndpoint(ref="L1", pin="1")],
            ),
        ],
    )

    isr = check_inter_sheet_nets_realized(arch, bom)
    assert not isr.ok
    # Exactly the 5 signal sheet-pin orphans KiCad ERC reported.
    assert len(isr.offenders) == 5
    assert sum("PWM_H" in o for o in isr.offenders) == 2
    assert sum("PWM_L" in o for o in isr.offenders) == 2
    assert sum("COIL_OUT" in o for o in isr.offenders) == 1
    # Power nets are not emitted as sheet pins -> never flagged here.
    assert not any(("VBUS" in o or "GND" in o or "+3V3" in o) for o in isr.offenders)

    sp = check_sheets_have_parts(arch, bom)
    assert not sp.ok
    assert len(sp.offenders) == 1 and "COIL DRIVER" in sp.offenders[0]


# ---------- §9.15 no dangling signal nets (SOIL_MOISTURE_BLE USB regression) ----------


def _soil_like_design(fixed: bool = False) -> tuple[Architecture, BOM]:
    """Mirror the SOIL_MOISTURE_BLE topology: a USB-C connector sheet and an
    ESP32 sheet whose USB D+/D- must join across the sheet boundary, alongside a
    correctly-declared ANALOG_OUT inter-sheet net and ordinary 2-pin local nets
    (CC1, EN) that must NOT be flagged.

    ``fixed=False`` reproduces the live bug: the USB data lines are four disjoint
    single-pin local nets (USB_DP_POWER/USB_DN_POWER on the connector,
    USB_DP_ESP32/USB_DN_ESP32 on the MCU) -- named inconsistently and never
    declared inter-sheet -- so each label connects to nothing.

    ``fixed=True`` carries each line as one declared inter-sheet net wired on
    both sides (USB_DP, USB_DN), which is how the design should have been wired.
    """
    inter = [
        InterSheetNet(
            name="GND",
            endpoints=[
                SheetPin(sheet=s, direction="bidirectional")
                for s in ("USB POWER", "ESP32", "CAP SENSOR")
            ],
        ),
        InterSheetNet(
            name="VCC_3V3",
            endpoints=[
                SheetPin(sheet="ESP32", direction="bidirectional"),
                SheetPin(sheet="CAP SENSOR", direction="bidirectional"),
            ],
        ),
        InterSheetNet(
            name="ANALOG_OUT",
            endpoints=[
                SheetPin(sheet="CAP SENSOR", direction="output"),
                SheetPin(sheet="ESP32", direction="input"),
            ],
        ),
    ]
    conns = [
        # Power (exempt) + a healthy 2-pin local net on the connector sheet.
        NetConnection(
            net_name="VBUS", sheet="USB POWER", endpoints=[PinEndpoint(ref="J1", pin="A4")]
        ),
        NetConnection(
            net_name="GND",
            sheet="USB POWER",
            endpoints=[PinEndpoint(ref="J1", pin="A1"), PinEndpoint(ref="R1", pin="2")],
        ),
        NetConnection(
            net_name="CC1",
            sheet="USB POWER",
            endpoints=[PinEndpoint(ref="J1", pin="A5"), PinEndpoint(ref="R1", pin="1")],
        ),
        # ESP32 sheet: power + a healthy 2-pin local net (EN) + the ANALOG_OUT
        # inter-sheet stub (single local pin, but joins across sheets -> OK).
        NetConnection(
            net_name="VCC_3V3",
            sheet="ESP32",
            endpoints=[PinEndpoint(ref="U2", pin="3"), PinEndpoint(ref="R3", pin="1")],
        ),
        NetConnection(
            net_name="EN",
            sheet="ESP32",
            endpoints=[PinEndpoint(ref="U2", pin="45"), PinEndpoint(ref="R3", pin="2")],
        ),
        NetConnection(
            net_name="ANALOG_OUT", sheet="ESP32", endpoints=[PinEndpoint(ref="U2", pin="8")]
        ),
        # CAP SENSOR sheet: the other ANALOG_OUT stub + power.
        NetConnection(
            net_name="ANALOG_OUT", sheet="CAP SENSOR", endpoints=[PinEndpoint(ref="J2", pin="3")]
        ),
        NetConnection(
            net_name="VCC_3V3", sheet="CAP SENSOR", endpoints=[PinEndpoint(ref="J2", pin="1")]
        ),
    ]
    if fixed:
        inter += [
            InterSheetNet(
                name="USB_DP",
                endpoints=[
                    SheetPin(sheet="USB POWER", direction="bidirectional"),
                    SheetPin(sheet="ESP32", direction="bidirectional"),
                ],
            ),
            InterSheetNet(
                name="USB_DN",
                endpoints=[
                    SheetPin(sheet="USB POWER", direction="bidirectional"),
                    SheetPin(sheet="ESP32", direction="bidirectional"),
                ],
            ),
        ]
        conns += [
            NetConnection(
                net_name="USB_DP", sheet="USB POWER", endpoints=[PinEndpoint(ref="J1", pin="A6")]
            ),
            NetConnection(
                net_name="USB_DP", sheet="ESP32", endpoints=[PinEndpoint(ref="U2", pin="24")]
            ),
            NetConnection(
                net_name="USB_DN", sheet="USB POWER", endpoints=[PinEndpoint(ref="J1", pin="A7")]
            ),
            NetConnection(
                net_name="USB_DN", sheet="ESP32", endpoints=[PinEndpoint(ref="U2", pin="23")]
            ),
        ]
    else:
        conns += [
            NetConnection(
                net_name="USB_DP_POWER",
                sheet="USB POWER",
                endpoints=[PinEndpoint(ref="J1", pin="A6")],
            ),
            NetConnection(
                net_name="USB_DN_POWER",
                sheet="USB POWER",
                endpoints=[PinEndpoint(ref="J1", pin="A7")],
            ),
            NetConnection(
                net_name="USB_DP_ESP32", sheet="ESP32", endpoints=[PinEndpoint(ref="U2", pin="24")]
            ),
            NetConnection(
                net_name="USB_DN_ESP32", sheet="ESP32", endpoints=[PinEndpoint(ref="U2", pin="23")]
            ),
        ]
    arch = Architecture(
        sheets=[
            Sheet(name="USB POWER", stem="USB_POWER", function="usb input"),
            Sheet(name="ESP32", stem="ESP32", function="mcu"),
            Sheet(name="CAP SENSOR", stem="CAP_SENSOR", function="sensor"),
        ],
        power_nets=["VBUS", "VCC_3V3", "GND"],
        inter_sheet_nets=inter,
    )
    bom = BOM(
        parts=[
            _part("J1", "USB POWER"),
            _part("R1", "USB POWER"),
            _part("U2", "ESP32"),
            _part("R3", "ESP32"),
            _part("J2", "CAP SENSOR"),
        ],
        connections=conns,
    )
    return arch, bom


def test_dangling_signal_nets_flags_soil_usb() -> None:
    """The live SOIL_MOISTURE_BLE failure: four disjoint single-pin USB nets,
    each a 'Label not connected to anything' ERC error."""
    arch, bom = _soil_like_design(fixed=False)
    r = check_no_dangling_signal_nets(arch, bom)
    assert not r.ok
    assert len(r.offenders) == 4
    # The A1 topology context may legitimately *mention* other net names
    # (declared inter-sheet list); the invariant is that no OTHER net is
    # flagged: each offender's lead clause must name one of the four USB
    # singletons, and the healthy nets never open an offender.
    leads = [o.split(" and is neither")[0] for o in r.offenders]
    blob = " ".join(leads)
    for net in ("USB_DP_POWER", "USB_DN_POWER", "USB_DP_ESP32", "USB_DN_ESP32"):
        assert f"net {net!r}" in blob
    # The healthy 2-pin local nets, the inter-sheet stub, and power must NOT
    # be flagged.
    for ok_net in ("CC1", "EN", "ANALOG_OUT", "VBUS", "VCC_3V3", "GND"):
        assert f"net {ok_net!r}" not in blob


# ---------- A1 (KC-VKUT5H): topology-safe §9.15 offender context ----------
#
# The context must (a) name candidate destination endpoints for a proven
# series far-side, (b) expose 74x245-style A/B channel mates, (c) NEVER
# suggest merging domain-split nets, and (d) keep the offender's identity
# (pin tokens) untouched: the lead clause's canonical REF.PIN remains the
# only match for _offender_identity's pin regex.

import re as _re

# EXACT production regex from stage_runtime._offender_identity.
_A1_CANON_PIN_RE = _re.compile(
    r"\b([A-Za-z]+[0-9]+[A-Za-z0-9_-]*)(?:\.|\s+pin\s+)([A-Za-z0-9~_+-]+)\b"
)

_A1_PINS = {
    "A:CONN": [("A6", "DP1", "biod"), ("A7", "DM1", "biod"), ("B12", "GND", "pwr")],
    "A:MCU": [
        ("14", "IO20/USB_D+", "bidir"),
        ("13", "IO19/USB_D-", "bidir"),
        ("16", "SPARE", "bidir"),
    ],
    "Device:R": [("1", "1", "passive"), ("2", "2", "passive")],
    "A:VAR": [("1", "1", "passive"), ("2", "2", "passive"), ("3", "3", "passive")],
    "esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1": [
        ("14", "IO20/USB_D+", "bidir"),
        ("36", "RXD0", "bidir"),
        ("37", "TXD0", "bidir"),
    ],
    "A:OCT": [
        ("1", "A3", "input"),
        ("2", "B3", "output"),
        ("20", "VCC", "pwr_in"),
    ],
}


def _a1_bom(conns, refs) -> BOM:
    parts = []
    for ref in refs:
        symbol = {
            "J1": "A:CONN",
            "U3": "A:MCU",
            "U5": "A:OCT",
            "RV1": "A:VAR",
        }.get(ref, "Device:R")
        parts.append(
            BomPart(
                ref=ref,
                value="x",
                symbol=symbol,
                footprint="Resistor_SMD:R_0402_1005Metric",
                sheet="MCU",
            )
        )
    return BOM(parts=parts, connections=conns)


def _a1_arch(inter=()) -> Architecture:
    sheets = [Sheet(name="MCU", stem="MCU", function="mcu")]
    if any("OTHER" in {e.sheet for e in n.endpoints} for n in inter):
        sheets.append(Sheet(name="OTHER", stem="OTHER", function="other"))
    return Architecture(
        sheets=sheets,
        power_nets=["+3V3", "GND"],
        inter_sheet_nets=list(inter),
    )


def _a1_offenders(monkeypatch, conns, refs, inter=()):
    monkeypatch.setattr(_sp, "lookup_pins", _fake_lookup(_A1_PINS))
    r = check_no_dangling_signal_nets(_a1_arch(inter), _a1_bom(conns, refs))
    return {o.split(" on sheet")[0].split("net ")[1].strip("'"): o for o in r.offenders}


def _a1_lead(off: str) -> str:
    return off.split(" -- ")[0]


def _assert_identity_safe(off: str) -> None:
    """Context adds zero REF.PIN / 'REF pin N' tokens beyond the lead clause."""
    canon = set(_A1_CANON_PIN_RE.findall(_a1_lead(off)))
    whole = set(_A1_CANON_PIN_RE.findall(off))
    assert whole == canon, f"context leaked pin tokens: {whole - canon}"


def test_a1_series_far_side_names_destination_candidates(monkeypatch) -> None:
    """The KC-VKUT5H USB case: R3.1 sits on USB_D_P with J1 and U3; R3.2
    dangles as USB_D_P_MCU. Feedback must list USB_D_P's non-series endpoints
    as identity-safe pin labels with functions, and demand the move keeps the
    two part terminals on different nets."""
    conns = [
        NetConnection(
            net_name="USB_D_P",
            sheet="MCU",
            endpoints=[
                PinEndpoint(ref="J1", pin="A6"),
                PinEndpoint(ref="U3", pin="14"),
                PinEndpoint(ref="R3", pin="1"),
            ],
        ),
        NetConnection(
            net_name="USB_D_P_MCU", sheet="MCU", endpoints=[PinEndpoint(ref="R3", pin="2")]
        ),
    ]
    offs = _a1_offenders(monkeypatch, conns, ["J1", "U3", "R3"])
    o = offs["USB_D_P_MCU"]
    _assert_identity_safe(o)
    assert "two-terminal series part" in o
    assert "pin 1 of R3" in o
    # Candidates sorted by (ref, pin); functions included even when the pin
    # NUMBER is trivial.
    assert "pin A6 of J1 (DP1)" in o
    assert "pin 14 of U3 (IO20/USB_D+)" in o
    assert o.index("pin A6 of J1") < o.index("pin 14 of U3")
    assert "Do not assume which side is source or destination" in o
    assert "Move the intended load/destination endpoint" not in o
    assert "never merge 'USB_D_P' with 'USB_D_P_MCU'" in o
    assert "never put both terminals of R3 on one net" in o


def test_a1_series_filters_wrong_fixed_signal_candidate(monkeypatch) -> None:
    """Board 783 shape: keep IO20, but never offer TXD0 as a D+ endpoint."""
    monkeypatch.setattr(_sp, "lookup_pins", _fake_lookup(_A1_PINS))
    bom = BOM(
        parts=[
            _bpart("J1", "A:CONN", sheet="MCU"),
            _bpart("R9", "Device:R", sheet="MCU"),
            _bpart(
                "U3",
                "esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1",
                sheet="MCU",
            ),
        ],
        connections=[
            NetConnection(
                net_name="USB_DP",
                sheet="MCU",
                endpoints=[
                    PinEndpoint(ref="J1", pin="A6"),
                    PinEndpoint(ref="R9", pin="1"),
                    PinEndpoint(ref="U3", pin="14"),
                    PinEndpoint(ref="U3", pin="37"),
                ],
            ),
            NetConnection(
                net_name="USB_DP_MCU",
                sheet="MCU",
                endpoints=[PinEndpoint(ref="R9", pin="2")],
            ),
        ],
    )
    result = check_no_dangling_signal_nets(_a1_arch(), bom)
    offender = next(o for o in result.offenders if "USB_DP_MCU" in _a1_lead(o))
    _assert_identity_safe(offender)
    candidates = offender.split("candidate endpoints on that net:", 1)[1].split(
        ". Keep each endpoint", 1
    )[0]
    assert "IO20" in candidates
    assert "TXD0" not in candidates
    assert "required fixed function: IO20" in offender
    assert "cannot carry either accepted name variant 'USB_DP' or 'USB_DP_MCU'" in offender
    assert "keeping the two terminals of R9 on different nets" in offender


def test_a1_series_candidate_filter_fails_open_when_pin_unresolved(monkeypatch) -> None:
    pinmap = {key: value for key, value in _A1_PINS.items() if "esp32-s3" not in key}
    monkeypatch.setattr(_sp, "lookup_pins", _fake_lookup(pinmap))
    bom = BOM(
        parts=[
            _bpart("R9", "Device:R", sheet="MCU"),
            _bpart(
                "U3",
                "esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1",
                sheet="MCU",
            ),
        ],
        connections=[
            NetConnection(
                net_name="USB_DP",
                sheet="MCU",
                endpoints=[
                    PinEndpoint(ref="R9", pin="1"),
                    PinEndpoint(ref="U3", pin="37"),
                ],
            ),
            NetConnection(
                net_name="USB_DP_MCU",
                sheet="MCU",
                endpoints=[PinEndpoint(ref="R9", pin="2")],
            ),
        ],
    )
    offender = check_no_dangling_signal_nets(_a1_arch(), bom).offenders[0]
    assert "candidate endpoints on that net: pin 37 of U3" in offender
    assert "Rejected wrong-function candidate" not in offender


def test_a1_series_guidance_requires_proven_two_pin_part(monkeypatch) -> None:
    """A three-pin RV part (prefix in _TWO_TERMINAL_REF_PREFIXES, but 3 pins)
    gets NO two-terminal series guidance (§9.17's exact pin-count invariant).
    """
    conns = [
        NetConnection(
            net_name="SIG_IN",
            sheet="MCU",
            endpoints=[PinEndpoint(ref="U3", pin="14"), PinEndpoint(ref="RV1", pin="1")],
        ),
        NetConnection(net_name="SIG_MID", sheet="MCU", endpoints=[PinEndpoint(ref="RV1", pin="2")]),
    ]
    offs = _a1_offenders(monkeypatch, conns, ["U3", "RV1"])
    o = offs["SIG_MID"]
    _assert_identity_safe(o)
    assert "two-terminal series part" not in o


def test_a1_series_guidance_omitted_when_other_terminal_ambiguous(monkeypatch) -> None:
    """R5.2 on two distinct nets (a §9.19 short) is ambiguous -> no guess."""
    conns = [
        NetConnection(
            net_name="SIG_DANGLE", sheet="MCU", endpoints=[PinEndpoint(ref="R5", pin="1")]
        ),
        NetConnection(
            net_name="SIG_A",
            sheet="MCU",
            endpoints=[PinEndpoint(ref="R5", pin="2"), PinEndpoint(ref="U3", pin="14")],
        ),
        NetConnection(net_name="SIG_B", sheet="MCU", endpoints=[PinEndpoint(ref="R5", pin="2")]),
    ]
    offs = _a1_offenders(monkeypatch, conns, ["U3", "R5"])
    o = offs["SIG_DANGLE"]
    _assert_identity_safe(o)
    assert "two-terminal series part" not in o


def test_a1_translator_channel_mates_exposed(monkeypatch) -> None:
    """The HUB75 74HCT245 case: dangling HUB75_CLK_5V (U5.2/B3) reports the
    same-sheet related net HUB75_CLK (U5.1/A3) AND names the channel mate
    pairing — with an explicit no-merge instruction."""
    conns = [
        NetConnection(
            net_name="HUB75_CLK",
            sheet="MCU",
            endpoints=[PinEndpoint(ref="U5", pin="1"), PinEndpoint(ref="R6", pin="1")],
        ),
        NetConnection(
            net_name="HUB75_CLK_5V", sheet="MCU", endpoints=[PinEndpoint(ref="U5", pin="2")]
        ),
    ]
    offs = _a1_offenders(monkeypatch, conns, ["U5", "R6"])
    o = offs["HUB75_CLK_5V"]
    _assert_identity_safe(o)
    assert "related net 'HUB75_CLK'" in o
    assert "pin 1 of U5 (A3)" in o
    assert "has its channel mate pin 2 of U5 (B3) on net 'HUB75_CLK_5V'" in o
    assert "do NOT merge 'HUB75_CLK_5V' with 'HUB75_CLK'" in o


def test_a1_negative_normalization(monkeypatch) -> None:
    """UART0/UART1, LED1/LED2, USB_D_P/USB_D_N never correlate; only an
    explicit listed domain suffix splits a related pair."""
    from kicraft.design.synthesis.validation import _net_domain_base

    assert _net_domain_base("UART0") == "UART0"  # numeric suffix never stripped
    assert _net_domain_base("UART0") != _net_domain_base("UART1")
    assert _net_domain_base("LED1") != _net_domain_base("LED2")
    assert _net_domain_base("USB_D_P") != _net_domain_base("USB_D_N")  # one-letter
    assert _net_domain_base("USB_D_P_MCU") == "USB_D_P"
    assert _net_domain_base("USB_D_N_MCU") == "USB_D_N"  # ONE suffix only
    assert _net_domain_base("HUB75_CLK_5V") == _net_domain_base("HUB75_CLK") == "HUB75_CLK"
    assert _net_domain_base("SIG_ISO") == "SIG"
    assert _net_domain_base("SIG") == "SIG"
    conns = [
        NetConnection(
            net_name="UART1_TX", sheet="MCU", endpoints=[PinEndpoint(ref="U3", pin="16")]
        ),
        NetConnection(
            net_name="UART0_TX",
            sheet="MCU",
            endpoints=[PinEndpoint(ref="U3", pin="14"), PinEndpoint(ref="U3", pin="13")],
        ),
    ]
    offs = _a1_offenders(monkeypatch, conns, ["U3"])
    o = offs["UART1_TX"]
    assert "related net" not in o


def test_a1_inter_sheet_list_and_determinism(monkeypatch) -> None:
    """Declared inter-sheet names are appended sorted, capped at eight; the
    whole offender string is deterministic across repeated calls; an unused
    GPIO still keeps the lead clause's three valid choices verbatim."""
    inter = [
        InterSheetNet(
            name=n,
            endpoints=[SheetPin(sheet=s, direction="bidirectional") for s in ("MCU", "OTHER")],
        )
        for n in ("ZZ_NET", "ALPHA_NET", "MB", "NB", "OB", "PB", "QB", "RB", "SB")
    ]
    conns = [
        NetConnection(
            net_name="ESP_UNUSED",
            sheet="MCU",
            endpoints=[PinEndpoint(ref="U3", pin="16")],
        )
    ]
    a = _a1_offenders(monkeypatch, conns, ["U3"], inter)
    b = _a1_offenders(monkeypatch, conns, ["U3"], inter)
    o = a["ESP_UNUSED"]
    assert o == b["ESP_UNUSED"]
    assert (
        "wire it to a second pin, mark it no_connect, or declare an "
        "inter-sheet net to carry it to another sheet" in o
    )
    assert "declared inter-sheet net names: ALPHA_NET, MB, NB, OB, PB, QB, RB, SB" in o
    assert "ZZ_NET" not in o
    assert "two-terminal series part" not in o
    assert "related net" not in o
    _assert_identity_safe(o)


def test_dangling_signal_nets_pass_when_usb_declared_intersheet() -> None:
    """Carrying USB_DP/USB_DN as declared inter-sheet nets wired on both sides
    clears §9.15 (and §9.14 stays happy)."""
    arch, bom = _soil_like_design(fixed=True)
    assert check_no_dangling_signal_nets(arch, bom).ok
    assert check_inter_sheet_nets_realized(arch, bom).ok


def test_dangling_signal_nets_ignore_power_and_intersheet() -> None:
    """A design whose only single-stub nets are power or declared inter-sheet
    (the §9.14 fixture) has nothing dangling."""
    arch, bom = _two_sheet_design()
    assert check_no_dangling_signal_nets(arch, bom).ok


def test_dangling_signal_nets_offender_names_the_pin() -> None:
    """The message is actionable: it names the orphaned pin so the wiring stage
    knows exactly what to fix."""
    arch, bom = _soil_like_design(fixed=False)
    offenders = check_no_dangling_signal_nets(arch, bom).offenders
    assert any("J1.A6" in o for o in offenders)
    assert any("U2.23" in o for o in offenders)


# ---------- §9.16-§9.18 semantic wiring checks ----------
#
# These cross-read each part's symbol pin NAMES against the polarity/role of the
# net every pin lands on, catching functionally-wrong netlists that are still
# ERC/DRC-legal (reversed power, a fuse shorted across itself, an antenna feed
# tied to GND). Pin data is faked via lookup_pins so the tests are hermetic.

import kicraft.design.synthesis.symbol_pinout as _sp


def _fake_lookup(pinmap):
    """lookup_pins stand-in over {symbol: [(num, name, electrical_type), ...]}."""
    from kicraft.design.synthesis.symbol_pinout import SymbolNotFoundError

    def _lookup(lib_id, *a, **k):
        if lib_id not in pinmap:
            raise SymbolNotFoundError(lib_id)
        return {
            "symbol": lib_id,
            "unit_count": 1,
            "pins": [
                {"number": n, "name": nm, "electrical_type": t} for (n, nm, t) in pinmap[lib_id]
            ],
        }

    return _lookup


def _bpart(ref, symbol, sheet="MCU"):
    return BomPart(ref=ref, value="x", symbol=symbol, footprint="Package_SO:SOIC-8", sheet=sheet)


def test_power_pin_polarity_flags_reversed_supply(monkeypatch) -> None:
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup({"Fake:MCU": [("1", "VDD", "power_in"), ("2", "VSS", "power_in")]}),
    )
    bom = BOM(
        parts=[_bpart("U1", "Fake:MCU")],
        connections=[
            NetConnection(
                net_name="GND", sheet="MCU", endpoints=[PinEndpoint(ref="U1", pin="1")]
            ),  # VDD -> GND
            NetConnection(
                net_name="+3V3", sheet="MCU", endpoints=[PinEndpoint(ref="U1", pin="2")]
            ),  # VSS -> +3V3
        ],
    )
    res = check_power_pin_polarity(bom)
    assert not res.ok
    assert len(res.offenders) == 2


def test_power_pin_polarity_passes_correct_supply(monkeypatch) -> None:
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup({"Fake:MCU": [("1", "VDD", "power_in"), ("2", "VSS", "power_in")]}),
    )
    bom = BOM(
        parts=[_bpart("U1", "Fake:MCU")],
        connections=[
            NetConnection(net_name="+3V3", sheet="MCU", endpoints=[PinEndpoint(ref="U1", pin="1")]),
            NetConnection(net_name="GND", sheet="MCU", endpoints=[PinEndpoint(ref="U1", pin="2")]),
        ],
    )
    assert check_power_pin_polarity(bom).ok


def _fake_two_supply_bom(positive_net: str, negative_net: str) -> BOM:
    return BOM(
        parts=[_bpart("U1", "Fake:OPAMP")],
        connections=[
            NetConnection(
                net_name=positive_net, sheet="MCU", endpoints=[PinEndpoint(ref="U1", pin="4")]
            ),
            NetConnection(
                net_name=negative_net, sheet="MCU", endpoints=[PinEndpoint(ref="U1", pin="11")]
            ),
        ],
    )


def _patch_fake_two_supply_lookup(monkeypatch) -> None:
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup(
            {
                "Fake:OPAMP": [
                    ("4", "VCC+", "power_in"),
                    ("11", "VCC-", "power_in"),
                ],
            }
        ),
    )


def test_power_pin_polarity_accepts_vcc_suffix_single_and_dual_supply(monkeypatch) -> None:
    _patch_fake_two_supply_lookup(monkeypatch)
    assert check_power_pin_polarity(_fake_two_supply_bom("+5V", "GND")).ok
    assert check_power_pin_polarity(_fake_two_supply_bom("+12V", "-12V")).ok


def test_power_pin_polarity_rejects_vcc_suffix_reversal(monkeypatch) -> None:
    _patch_fake_two_supply_lookup(monkeypatch)
    res = check_power_pin_polarity(_fake_two_supply_bom("GND", "+5V"))
    assert not res.ok
    assert len(res.offenders) == 2
    assert any("U1.4" in offender and "ground net" in offender for offender in res.offenders)
    assert any("U1.11" in offender and "positive rail" in offender for offender in res.offenders)


@pytest.mark.parametrize("negative_net", ["VSS", "VEE", "-12V"])
def test_power_pin_polarity_rejects_positive_pin_on_negative_net(
    monkeypatch,
    negative_net: str,
) -> None:
    _patch_fake_two_supply_lookup(monkeypatch)
    res = check_power_pin_polarity(_fake_two_supply_bom(negative_net, "+5V"))
    assert not res.ok
    assert any(
        "U1.4" in offender and negative_net in offender and "negative rail" in offender
        for offender in res.offenders
    )
    assert any("U1.11" in offender and "positive rail" in offender for offender in res.offenders)


def test_power_pin_polarity_ignores_differential_input(monkeypatch) -> None:
    # A differential analog input (VIN-/VINP) must NOT be read as a supply.
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup({"Fake:ADC": [("1", "VIN-", "input"), ("2", "VINP", "input")]}),
    )
    bom = BOM(
        parts=[_bpart("U2", "Fake:ADC")],
        connections=[
            NetConnection(net_name="GND", sheet="MCU", endpoints=[PinEndpoint(ref="U2", pin="1")]),
            NetConnection(net_name="+3V3", sheet="MCU", endpoints=[PinEndpoint(ref="U2", pin="2")]),
        ],
    )
    assert check_power_pin_polarity(bom).ok


def test_two_terminal_self_short_flags_fuse_across_one_net(monkeypatch) -> None:
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup({"Device:Fuse": [("1", "~", "passive"), ("2", "~", "passive")]}),
    )
    bom = BOM(
        parts=[_bpart("F1", "Device:Fuse", sheet="POWER")],
        connections=[
            NetConnection(
                net_name="VIN",
                sheet="POWER",
                endpoints=[PinEndpoint(ref="F1", pin="1"), PinEndpoint(ref="F1", pin="2")],
            ),
        ],
    )
    res = check_two_terminal_self_short(bom)
    assert not res.ok
    assert len(res.offenders) == 1


def test_two_terminal_self_short_passes_series_part(monkeypatch) -> None:
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup({"Device:Fuse": [("1", "~", "passive"), ("2", "~", "passive")]}),
    )
    bom = BOM(
        parts=[_bpart("F1", "Device:Fuse", sheet="POWER")],
        connections=[
            NetConnection(
                net_name="VIN", sheet="POWER", endpoints=[PinEndpoint(ref="F1", pin="1")]
            ),
            NetConnection(
                net_name="VOUT", sheet="POWER", endpoints=[PinEndpoint(ref="F1", pin="2")]
            ),
        ],
    )
    assert check_two_terminal_self_short(bom).ok


def test_rf_feed_isolation_flags_antenna_feed_on_gnd(monkeypatch) -> None:
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup({"Fake:ANT": [("1", "FEED", "passive"), ("2", "GND", "passive")]}),
    )
    bom = BOM(
        parts=[_bpart("ANT1", "Fake:ANT", sheet="RF")],
        connections=[
            NetConnection(
                net_name="GND", sheet="RF", endpoints=[PinEndpoint(ref="ANT1", pin="1")]
            ),  # FEED -> GND
            NetConnection(net_name="GND", sheet="RF", endpoints=[PinEndpoint(ref="ANT1", pin="2")]),
        ],
    )
    res = check_rf_feed_isolation(bom)
    assert not res.ok
    assert len(res.offenders) == 1


def test_rf_feed_isolation_passes_feed_on_rf_net(monkeypatch) -> None:
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup({"Fake:ANT": [("1", "FEED", "passive"), ("2", "GND", "passive")]}),
    )
    bom = BOM(
        parts=[_bpart("ANT1", "Fake:ANT", sheet="RF")],
        connections=[
            NetConnection(
                net_name="ANT_FEED", sheet="RF", endpoints=[PinEndpoint(ref="ANT1", pin="1")]
            ),
            NetConnection(net_name="GND", sheet="RF", endpoints=[PinEndpoint(ref="ANT1", pin="2")]),
        ],
    )
    assert check_rf_feed_isolation(bom).ok


# ---------- §9.19 single net per pin + §9.20 family contracts (Layer 2) ----------

from kicraft.design.synthesis.validation import (  # noqa: E402
    check_family_wiring_contracts,
    check_single_net_per_pin,
)


def test_single_net_per_pin_flags_pin_on_two_nets() -> None:
    # DRV8833-style: VM pin (12) listed on both VBAT and VCP_VM shorts them
    # (and removes the charge-pump cap). No symbol lookup needed.
    bom = BOM(
        parts=[_bpart("U4", "drv8833:DRV8833", sheet="DRV")],
        connections=[
            NetConnection(
                net_name="VBAT", sheet="DRV", endpoints=[PinEndpoint(ref="U4", pin="12")]
            ),
            NetConnection(
                net_name="VCP_VM",
                sheet="DRV",
                endpoints=[PinEndpoint(ref="U4", pin="11"), PinEndpoint(ref="U4", pin="12")],
            ),
        ],
    )
    res = check_single_net_per_pin(bom)
    assert not res.ok
    assert len(res.offenders) == 1
    assert "U4.12" in res.offenders[0]


def test_single_net_per_pin_passes_clean_wiring() -> None:
    bom = BOM(
        parts=[_bpart("U4", "drv8833:DRV8833", sheet="DRV")],
        connections=[
            NetConnection(
                net_name="VBAT", sheet="DRV", endpoints=[PinEndpoint(ref="U4", pin="12")]
            ),
            NetConnection(
                net_name="VCP_VM", sheet="DRV", endpoints=[PinEndpoint(ref="U4", pin="11")]
            ),
        ],
    )
    assert check_single_net_per_pin(bom).ok


def test_single_net_per_pin_allows_repeated_name() -> None:
    # The same net_name appearing in two connections (different pins, one each)
    # is not a short -- only DISTINCT names on ONE pin are flagged.
    bom = BOM(
        parts=[_bpart("U1", "Fake:X", sheet="MCU")],
        connections=[
            NetConnection(net_name="SIG", sheet="MCU", endpoints=[PinEndpoint(ref="U1", pin="1")]),
            NetConnection(net_name="SIG", sheet="MCU", endpoints=[PinEndpoint(ref="U1", pin="2")]),
        ],
    )
    assert check_single_net_per_pin(bom).ok


def test_family_contract_flags_flash_supply_on_data_net(monkeypatch) -> None:
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup(
            {
                "w25q:W25Q16": [
                    ("8", "VCC", "power_in"),
                    ("4", "GND", "power_in"),
                    ("5", "DI/IO0", "bidirectional"),
                    ("6", "CLK", "input"),
                ]
            }
        ),
    )
    bom = BOM(
        parts=[_bpart("U2", "w25q:W25Q16", sheet="MCU")],
        connections=[
            NetConnection(
                net_name="QSPI_SD0",
                sheet="MCU",  # VCC scrambled onto a data net
                endpoints=[PinEndpoint(ref="U2", pin="8")],
            ),
            NetConnection(net_name="GND", sheet="MCU", endpoints=[PinEndpoint(ref="U2", pin="4")]),
            NetConnection(
                net_name="QSPI_SD1", sheet="MCU", endpoints=[PinEndpoint(ref="U2", pin="5")]
            ),
            NetConnection(
                net_name="QSPI_SCLK", sheet="MCU", endpoints=[PinEndpoint(ref="U2", pin="6")]
            ),
        ],
    )
    res = check_family_wiring_contracts(bom)
    assert not res.ok
    assert any("U2.8" in o for o in res.offenders)


def test_family_contract_flags_flash_data_on_rail(monkeypatch) -> None:
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup(
            {
                "w25q:W25Q16": [
                    ("8", "VCC", "power_in"),
                    ("4", "GND", "power_in"),
                    ("5", "DI/IO0", "bidirectional"),
                ]
            }
        ),
    )
    bom = BOM(
        parts=[_bpart("U2", "w25q:W25Q16", sheet="MCU")],
        connections=[
            NetConnection(net_name="+3V3", sheet="MCU", endpoints=[PinEndpoint(ref="U2", pin="8")]),
            NetConnection(net_name="GND", sheet="MCU", endpoints=[PinEndpoint(ref="U2", pin="4")]),
            NetConnection(
                net_name="+3V3",
                sheet="MCU",  # IO0 data line tied to the rail
                endpoints=[PinEndpoint(ref="U2", pin="5")],
            ),
        ],
    )
    res = check_family_wiring_contracts(bom)
    assert not res.ok
    assert any("U2.5" in o for o in res.offenders)


def test_family_contract_passes_correct_flash(monkeypatch) -> None:
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup(
            {
                "w25q:W25Q16": [
                    ("8", "VCC", "power_in"),
                    ("4", "GND", "power_in"),
                    ("5", "DI/IO0", "bidirectional"),
                    ("6", "CLK", "input"),
                    ("1", "~{CS}", "input"),
                ]
            }
        ),
    )
    bom = BOM(
        parts=[_bpart("U2", "w25q:W25Q16", sheet="MCU")],
        connections=[
            NetConnection(net_name="+3V3", sheet="MCU", endpoints=[PinEndpoint(ref="U2", pin="8")]),
            NetConnection(net_name="GND", sheet="MCU", endpoints=[PinEndpoint(ref="U2", pin="4")]),
            NetConnection(
                net_name="QSPI_SD0", sheet="MCU", endpoints=[PinEndpoint(ref="U2", pin="5")]
            ),
            NetConnection(
                net_name="QSPI_SCLK", sheet="MCU", endpoints=[PinEndpoint(ref="U2", pin="6")]
            ),
            NetConnection(
                net_name="QSPI_CSn", sheet="MCU", endpoints=[PinEndpoint(ref="U2", pin="1")]
            ),
        ],
    )
    assert check_family_wiring_contracts(bom).ok


def test_family_contract_flags_can_rs_on_rail(monkeypatch) -> None:
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup(
            {
                "sn65hvd230:SN65HVD230": [
                    ("3", "VCC", "power_in"),
                    ("2", "GND", "power_in"),
                    ("8", "RS", "input"),
                ]
            }
        ),
    )
    bom = BOM(
        parts=[_bpart("U3", "sn65hvd230:SN65HVD230", sheet="CAN")],
        connections=[
            NetConnection(net_name="+3V3", sheet="CAN", endpoints=[PinEndpoint(ref="U3", pin="3")]),
            NetConnection(net_name="GND", sheet="CAN", endpoints=[PinEndpoint(ref="U3", pin="2")]),
            NetConnection(
                net_name="+3V3",
                sheet="CAN",  # RS high = standby (wrong)
                endpoints=[PinEndpoint(ref="U3", pin="8")],
            ),
        ],
    )
    res = check_family_wiring_contracts(bom)
    assert not res.ok
    assert any("U3.8" in o for o in res.offenders)


# ---------- A2 (KC-VKUT5H): ESP32-S3 native-USB fixed-function assignment ----

_ESP32S3_PINS = {
    "esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1": [
        ("13", "IO19/USB_D-", "bidirectional"),
        ("14", "IO20/USB_D+", "bidirectional"),
        ("19", "IO11", "bidirectional"),
        ("20", "IO12", "bidirectional"),
        ("21", "IO13", "bidirectional"),
        ("22", "IO14", "bidirectional"),
        ("37", "TXD0", "bidirectional"),
    ],
    "esp32s3-mini:ESP32S3-MINI-1": [
        ("13", "IO19", "bidirectional"),
        ("14", "IO20", "bidirectional"),
        ("19", "IO11", "bidirectional"),
    ],
    "esp32:ESP32-WROOM-32": [("13", "IO19", "bidirectional")],
    "ch340g:CH340G": [("5", "D+", "bidirectional"), ("6", "D-", "bidirectional")],
}


def _usb_bom(ref, symbol, sheet, pins_by_net):
    return BOM(
        parts=[
            _bpart(ref, symbol, sheet=sheet),
            _bpart("R7", "Device:R", sheet=sheet),
        ],
        connections=[
            NetConnection(
                net_name=net,
                sheet=sheet,
                endpoints=[
                    PinEndpoint(ref=ref, pin=p),
                    PinEndpoint(ref="R7", pin=str(i + 1)),
                ],
            )
            for i, (net, p) in enumerate(pins_by_net.items())
        ],
    )


def _a2(monkeypatch, symbol, pins_by_net, ref="U3"):
    monkeypatch.setattr(_sp, "lookup_pins", _fake_lookup(_ESP32S3_PINS))
    return check_family_wiring_contracts(_usb_bom(ref, symbol, "MCU", pins_by_net))


def test_a2_correct_native_usb_passes_with_aliases(monkeypatch) -> None:
    """D+ on IO20, D- on IO19 — including the WROOM alias names
    IO20/USB_D+ / IO19/USB_D- — passes."""
    res = _a2(
        monkeypatch,
        "esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1",
        {"USB_D_P_MCU": "14", "USB_D_N_MCU": "13"},
    )
    assert res.ok, res.offenders
    res = _a2(monkeypatch, "esp32s3-mini:ESP32S3-MINI-1", {"USB_DP": "14", "USB_DN": "13"})
    assert res.ok, res.offenders


def test_a2_swapped_polarity_fails(monkeypatch) -> None:
    res = _a2(
        monkeypatch,
        "esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1",
        {"USB_D_P": "13", "USB_D_N": "14"},
    )
    assert not res.ok
    blob = " ".join(res.offenders)
    assert "pin 13 of U3 (IO19/USB_D-)" in blob  # identity-safe actual label
    assert "IO20" in blob and "D+" in blob
    assert "pin 14 of U3 (IO20/USB_D+)" in blob and "IO19" in blob
    # The feedback names the concrete target pin + the swap, not just a
    # function to hunt for.
    assert (
        "the correct endpoint is pin 14 of U3 (IO20/USB_D+), currently on net "
        "'USB_D_N' (swap the two)" in blob
    )


def test_a2_same_signal_variants_remove_wrong_pin_without_merge(monkeypatch) -> None:
    """Attempts 4/5: both names mean D+, so swapping them cannot fix TXD0."""
    res = _a2(
        monkeypatch,
        "esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1",
        {"USB_DP": "37", "USB_DP_MCU": "14"},
    )
    assert not res.ok
    assert len(res.offenders) == 1
    offender = res.offenders[0]
    assert "remove pin 37 of U3 (TXD0) from every accepted name" in offender
    assert "swap the two" not in offender
    assert "do not merge 'USB_DP' with 'USB_DP_MCU'" in offender
    assert "keep any proven series-part terminals on different nets" in offender
    assert "otherwise mark it no_connect" in offender
    signature = _re.findall(
        r"\b([A-Za-z]+\d+[A-Za-z0-9_-]*)(?:\.|\s+pin\s+)([A-Za-z0-9~_+-]+)\b",
        offender,
    )
    assert signature == []


def test_a2_frozen_candidates_wrong_gpios_fail(monkeypatch) -> None:
    """Attempts 1-2 (IO11/IO12) and attempt 3 (IO11+IO13 / IO12+IO14) of
    KC-VKUT5H — every wrong-function binding fires."""
    for pins in (
        {"USB_D_P": "19", "USB_D_N": "20"},  # IO11 / IO12
        {"USB_D_P": "19", "USB_D_N": "21"},  # IO11 / IO13
        {"USB_D+": "20", "USB_D-": "22"},  # IO12 / IO14
    ):
        res = _a2(monkeypatch, "esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1", pins)
        assert not res.ok, pins
        assert all("native USB" in o for o in res.offenders)


def test_a2_classifies_suffixed_net_names(monkeypatch) -> None:
    """Every exact differential form with one known domain suffix classifies;
    near-miss names never do (no loose substring matching)."""
    for net in ("USB_DP_ESP32", "USB_D_P_POWER", "USB_D-5V", "USB_DN_HV"):
        res = _a2(monkeypatch, "esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1", {net: "19"})
        assert not res.ok, net
    for net in ("USB_DPH", "USB_P", "D_P", "USBX_D_P", "USB_D_Q", "USB_D"):
        res = _a2(monkeypatch, "esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1", {net: "19"})
        assert res.ok, net


def test_a2_fails_open_for_unrelated_families(monkeypatch) -> None:
    """CH340's D+/D- pins and a classic ESP32 (non-S3) are outside the
    contract; no USB-named net on an S3 part is never inferred."""
    res = _a2(monkeypatch, "ch340g:CH340G", {"USB_D_P": "5", "USB_D_N": "6"})
    assert res.ok, res.offenders
    res = _a2(monkeypatch, "esp32:ESP32-WROOM-32", {"USB_D_P": "13"})
    assert res.ok, res.offenders
    res = _a2(monkeypatch, "esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1", {"SPI_CLK": "19"})
    assert res.ok, res.offenders


# ---------- §9.21 MCU first-flash / programming path (advisory) ----------

from kicraft.design.synthesis.validation import (  # noqa: E402
    check_mcu_programming_path,
)


def test_mcu_prog_path_flags_esp32_boot_strap_hard_tied(monkeypatch) -> None:
    # #12 esp32-s3: IO0 hard-tied to +3V3 -> cannot enter download mode.
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup(
            {
                "esp32:ESP32-S3": [
                    ("1", "VDD", "power_in"),
                    ("2", "GND", "power_in"),
                    ("3", "IO0", "bidirectional"),
                ]
            }
        ),
    )
    bom = BOM(
        parts=[_bpart("U1", "esp32:ESP32-S3")],
        connections=[
            NetConnection(net_name="+3V3", sheet="MCU", endpoints=[PinEndpoint(ref="U1", pin="1")]),
            NetConnection(net_name="GND", sheet="MCU", endpoints=[PinEndpoint(ref="U1", pin="2")]),
            NetConnection(net_name="+3V3", sheet="MCU", endpoints=[PinEndpoint(ref="U1", pin="3")]),
        ],
    )
    res = check_mcu_programming_path(bom)
    assert not res.ok and any("U1" in o and "IO0/GPIO0" in o for o in res.offenders)


def test_mcu_prog_path_passes_esp32_drivable_strap(monkeypatch) -> None:
    # IO0 on a signal net (a boot button/strap can pull it low) -> OK.
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup(
            {
                "esp32:ESP32-S3": [
                    ("3", "IO0", "bidirectional"),
                ]
            }
        ),
    )
    bom = BOM(
        parts=[
            _bpart("U1", "esp32:ESP32-S3"),
            BomPart(
                ref="SW1",
                value="boot",
                symbol="Switch:SW_Push",
                footprint="Button_Switch_SMD:SW_SPST",
                sheet="MCU",
            ),
        ],
        connections=[
            NetConnection(
                net_name="BOOT",
                sheet="MCU",
                endpoints=[PinEndpoint(ref="U1", pin="3"), PinEndpoint(ref="SW1", pin="1")],
            ),
        ],
    )
    assert check_mcu_programming_path(bom).ok


def test_mcu_prog_path_flags_rp2040_no_swd_no_button(monkeypatch) -> None:
    # #10 rp2040-min: SWD no-connect + no BOOTSEL button -> unprogrammable.
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup(
            {
                "rp2040:RP2040": [
                    ("1", "VDD", "power_in"),
                    ("2", "SWCLK", "input"),
                    ("3", "SWDIO", "bidirectional"),
                ]
            }
        ),
    )
    bom = BOM(
        parts=[_bpart("U1", "rp2040:RP2040")],
        connections=[
            NetConnection(net_name="+3V3", sheet="MCU", endpoints=[PinEndpoint(ref="U1", pin="1")]),
        ],
        no_connect_pins=[PinEndpoint(ref="U1", pin="2"), PinEndpoint(ref="U1", pin="3")],
    )
    res = check_mcu_programming_path(bom)
    assert not res.ok and any("SWD" in o for o in res.offenders)


def test_mcu_prog_path_passes_rp2040_with_swd_broken_out(monkeypatch) -> None:
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup(
            {
                "rp2040:RP2040": [
                    ("2", "SWCLK", "input"),
                    ("3", "SWDIO", "bidirectional"),
                ]
            }
        ),
    )
    bom = BOM(
        parts=[
            _bpart("U1", "rp2040:RP2040"),
            BomPart(
                ref="J1",
                value="SWD",
                symbol="Connector_Generic:Conn_01x04",
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical",
                sheet="MCU",
            ),
        ],
        connections=[
            NetConnection(
                net_name="SWCLK",
                sheet="MCU",
                endpoints=[PinEndpoint(ref="U1", pin="2"), PinEndpoint(ref="J1", pin="4")],
            ),
            NetConnection(
                net_name="SWDIO",
                sheet="MCU",
                endpoints=[PinEndpoint(ref="U1", pin="3"), PinEndpoint(ref="J1", pin="2")],
            ),
            NetConnection(net_name="GND", sheet="MCU", endpoints=[PinEndpoint(ref="J1", pin="3")]),
            NetConnection(net_name="+3V3", sheet="MCU", endpoints=[PinEndpoint(ref="J1", pin="1")]),
        ],
    )
    assert check_mcu_programming_path(bom).ok


def test_mcu_prog_path_passes_rp2040_with_bootsel_button(monkeypatch) -> None:
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup(
            {
                "rp2040:RP2040": [
                    ("2", "SWCLK", "input"),
                    ("3", "SWDIO", "bidirectional"),
                    ("4", "~{QSPI_SS}", "output"),
                ]
            }
        ),
    )
    bom = BOM(
        parts=[
            _bpart("U1", "rp2040:RP2040"),
            BomPart(
                ref="SW1",
                value="boot",
                symbol="Switch:SW_Push",
                footprint="Button_Switch_SMD:SW_SPST",
                sheet="MCU",
            ),
        ],
        connections=[
            NetConnection(net_name="GND", sheet="MCU", endpoints=[PinEndpoint(ref="SW1", pin="1")]),
            NetConnection(
                net_name="QSPI_CS",
                sheet="MCU",
                endpoints=[PinEndpoint(ref="U1", pin="4"), PinEndpoint(ref="SW1", pin="2")],
            ),
        ],
        no_connect_pins=[PinEndpoint(ref="U1", pin="2"), PinEndpoint(ref="U1", pin="3")],
    )
    assert check_mcu_programming_path(bom).ok


def test_mcu_prog_path_ignores_non_mcu_parts(monkeypatch) -> None:
    # A plain regulator is not an MCU -> never flagged.
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup(
            {
                "Regulator:AMS1117": [
                    ("1", "GND", "power_in"),
                    ("2", "VOUT", "power_out"),
                    ("3", "VIN", "power_in"),
                ]
            }
        ),
    )
    bom = BOM(
        parts=[_bpart("U1", "Regulator:AMS1117")],
        connections=[
            NetConnection(net_name="GND", sheet="MCU", endpoints=[PinEndpoint(ref="U1", pin="1")]),
        ],
    )
    assert check_mcu_programming_path(bom).ok


def test_mcu_prog_path_flags_generic_mcu_unconnected_swd(monkeypatch) -> None:
    monkeypatch.setattr(
        _sp,
        "lookup_pins",
        _fake_lookup(
            {
                "MCU_ST:STM32F030": [
                    ("1", "VDD", "power_in"),
                    ("2", "SWCLK", "input"),
                    ("3", "SWDIO", "bidirectional"),
                ]
            }
        ),
    )
    bom = BOM(
        parts=[_bpart("U1", "MCU_ST:STM32F030")],
        connections=[
            NetConnection(net_name="+3V3", sheet="MCU", endpoints=[PinEndpoint(ref="U1", pin="1")]),
        ],
        no_connect_pins=[PinEndpoint(ref="U1", pin="2"), PinEndpoint(ref="U1", pin="3")],
    )
    res = check_mcu_programming_path(bom)
    assert not res.ok and any("SWD/JTAG" in o for o in res.offenders)


# ---------- §9.22 breakout / adapter intent (advisory) ----------

from kicraft.design.synthesis.validation import (  # noqa: E402
    check_breakout_connectivity,
)
from kicraft.design.models import IntentSlot  # noqa: E402


def _conn(ref, sheet="IO"):
    return BomPart(
        ref=ref,
        value="hdr",
        symbol="Connector:Conn_01x04",
        footprint="Connector_PinHeader:PinHeader_1x04",
        sheet=sheet,
    )


def test_breakout_flags_unbridged_connectors() -> None:
    # #11 fpc-breakout: two connectors, no net spans both -> intent undone.
    intent = IntentSlot(goal="A simple FPC-to-header breakout board")
    bom = BOM(
        parts=[_conn("J1"), _conn("J2")],
        connections=[
            NetConnection(net_name="A", sheet="IO", endpoints=[PinEndpoint(ref="J1", pin="1")]),
            NetConnection(net_name="B", sheet="IO", endpoints=[PinEndpoint(ref="J2", pin="1")]),
        ],
    )
    res = check_breakout_connectivity(intent, bom)
    assert not res.ok and any("J1" in o and "J2" in o for o in res.offenders)


def test_breakout_passes_when_a_net_bridges() -> None:
    intent = IntentSlot(goal="A USB breakout / adapter")
    bom = BOM(
        parts=[_conn("J1"), _conn("J2")],
        connections=[
            NetConnection(
                net_name="D+",
                sheet="IO",
                endpoints=[PinEndpoint(ref="J1", pin="1"), PinEndpoint(ref="J2", pin="1")],
            ),
        ],
    )
    assert check_breakout_connectivity(intent, bom).ok


def test_breakout_skips_non_breakout_brief() -> None:
    intent = IntentSlot(goal="A 3.3V buck regulator board")
    bom = BOM(
        parts=[_conn("J1"), _conn("J2")],
        connections=[
            NetConnection(net_name="A", sheet="IO", endpoints=[PinEndpoint(ref="J1", pin="1")]),
            NetConnection(net_name="B", sheet="IO", endpoints=[PinEndpoint(ref="J2", pin="1")]),
        ],
    )
    assert check_breakout_connectivity(intent, bom).ok  # not a breakout -> not judged


def test_breakout_skips_single_connector() -> None:
    intent = IntentSlot(goal="A sensor breakout board")
    bom = BOM(
        parts=[_conn("J1")],
        connections=[
            NetConnection(net_name="A", sheet="IO", endpoints=[PinEndpoint(ref="J1", pin="1")]),
        ],
    )
    assert check_breakout_connectivity(intent, bom).ok  # <2 connectors -> not judged


# ---------- §9.25 capacitor symbol/footprint polarity consistency ----------


def _cap(ref: str, symbol: str, footprint: str) -> BomPart:
    return BomPart(ref=ref, value="10uF", symbol=symbol, footprint=footprint, sheet="PWR")


def test_cap_polarity_flags_nonpolar_symbol_on_polarized_footprint() -> None:
    # The KC-U2VAA8 defect: a film cap (Device:C) given an electrolytic can.
    bom = BOM(parts=[_cap("C1", "Device:C", "Capacitor_THT:CP_Radial_D12.5mm_P7.50mm")])
    r = check_capacitor_polarity_consistency(bom)
    assert not r.ok
    assert r.offenders and "C1" in r.offenders[0]


def test_cap_polarity_flags_polar_symbol_on_nonpolarized_footprint() -> None:
    bom = BOM(parts=[_cap("C2", "Device:CP", "Capacitor_SMD:C_0805_2012Metric")])
    assert not check_capacitor_polarity_consistency(bom).ok


def test_cap_polarity_flags_tantalum_footprint_on_nonpolar_symbol() -> None:
    bom = BOM(parts=[_cap("C3", "Device:C", "Capacitor_Tantalum_SMD:CP_EIA-3216-18")])
    assert not check_capacitor_polarity_consistency(bom).ok


def test_cap_polarity_passes_matching_nonpolarized() -> None:
    bom = BOM(parts=[_cap("C4", "Device:C", "Capacitor_SMD:C_0805_2012Metric")])
    assert check_capacitor_polarity_consistency(bom).ok


def test_cap_polarity_passes_matching_polarized() -> None:
    bom = BOM(parts=[_cap("C5", "Device:CP", "Capacitor_THT:CP_Radial_D8.0mm_P3.50mm")])
    assert check_capacitor_polarity_consistency(bom).ok


def test_cap_polarity_ignores_non_capacitors() -> None:
    # Resistors, inductors, diodes, crystals, MCUs must never trip the gate.
    bom = BOM(
        parts=[
            _part("R1", "PWR"),
            _cap("L1", "Device:L", "Inductor_THT:L_Radial_D21.0mm"),
            _cap("D1", "Device:D", "Diode_SMD:D_SOD-123"),
            _cap("Y1", "Device:Crystal", "Crystal:Crystal_SMD_3225-4Pin"),
            _cap("U1", "MCU_ST:STM32", "Package_QFP:LQFP-48"),
        ]
    )
    assert check_capacitor_polarity_consistency(bom).ok


def test_cap_polarity_skips_unrecognized_names() -> None:
    # Custom/vendored names that don't follow the C/CP convention are skipped,
    # not guessed (no false positive).
    bom = BOM(parts=[_cap("C6", "vendored:FilmBox", "vendored:film_5mm")])
    assert check_capacitor_polarity_consistency(bom).ok


# --------------------------------------------------------------------------- #
# §9.29 MCU programming access (hard gate): part presence + UPDI reachability
# --------------------------------------------------------------------------- #
# KC-HN59RJ shipped an ATtiny412 whose UPDI net had a pullup but no header or
# test pad -- electrically clean, physically unprogrammable. pid 592 shipped an
# STM32 with no programming part at all. The §9.21 advisory covers neither
# (UPDI-blind, and non-blocking); §9.29 makes both deterministic and hard.

from kicraft.design.synthesis.validation import check_mcu_programming_access


def _jpart(ref, value, symbol="Connector:Conn_01x03", sheet="MCU"):
    return BomPart(
        ref=ref,
        value=value,
        symbol=symbol,
        footprint="Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical",
        sheet=sheet,
    )


ATTINY = "attiny412:ATTINY412-SSNR"
ATTINY_PINS = {
    ATTINY: [
        ("1", "VDD", "power_in"),
        ("8", "GND", "power_in"),
        ("6", "UPDI/PA0", "bidirectional"),
        ("4", "PA6", "bidirectional"),
    ]
}


def test_prog_access_ok_without_mcu() -> None:
    bom = BOM(
        parts=[
            _bpart("J1", "jst:JST"),
        ],
        connections=[],
    )
    assert check_mcu_programming_access(bom).ok


def test_prog_access_fails_mcu_with_no_access_part() -> None:
    # The KC-HN59RJ BOM shape: MCU + power JST only. Must fail at BOM commit
    # (no connections yet) with the MCU as offender and actionable text.
    bom = BOM(
        parts=[
            _bpart("U1", ATTINY),
            _jpart("J1", "JST-PH 2-pin SMD", symbol="jst-ph-2p:S2B-PH-SM4-TB"),
        ],
        connections=[],
    )
    res = check_mcu_programming_access(bom)
    assert not res.ok
    assert res.offenders and "U1" in res.offenders[0]
    assert "UPDI" in res.message and "test" in res.message.lower()


def test_prog_access_part_presence_satisfied_by_updi_header() -> None:
    bom = BOM(parts=[_bpart("U1", ATTINY), _jpart("J2", "UPDI header 1x03")], connections=[])
    assert check_mcu_programming_access(bom).ok


def test_prog_access_part_presence_satisfied_by_test_pads() -> None:
    tp = BomPart(
        ref="TP1",
        value="UPDI pad",
        symbol="Connector:TestPoint",
        footprint="TestPoint:TestPoint_Pad_D1.5mm",
        sheet="MCU",
    )
    bom = BOM(parts=[_bpart("U1", ATTINY), tp], connections=[])
    assert check_mcu_programming_access(bom).ok


def test_prog_access_part_presence_usb_alone_insufficient_for_esp32() -> None:
    # Family strap rule (self-eval 2026-07-19 run_30): a bare USB connector is
    # not a workable ESP32 download-mode story -- entering the bootloader
    # needs BOOT+EN buttons, strap test pads, or a USB-UART bridge whose
    # DTR/RTS auto-reset drives the straps. The bridge variant passes.
    usb = BomPart(
        ref="J1",
        value="USB-C receptacle",
        symbol="usbc:TYPE-C-16P",
        footprint="usbc:HRO-TYPE-C-16P",
        sheet="POWER",
    )
    mcu = _bpart("U1", "esp32-s3-mini-1:ESP32-S3-MINI-1")
    res = check_mcu_programming_access(BOM(parts=[mcu, usb], connections=[]))
    assert not res.ok
    assert "download mode" in res.offenders[0]
    bridge = BomPart(
        ref="U2",
        value="CH340C USB-UART bridge",
        symbol="Device:R",
        footprint="Resistor_SMD:R_0402_1005Metric",
        sheet="POWER",
    )
    assert check_mcu_programming_access(BOM(parts=[mcu, usb, bridge], connections=[])).ok


def test_prog_access_updi_wired_to_pullup_only_fails(monkeypatch) -> None:
    # The exact observed failure: UPDI net exists (pullup R1) but never reaches
    # the header -- wiring commit must reject with a wire-it instruction.
    monkeypatch.setattr(_sp, "lookup_pins", _fake_lookup(ATTINY_PINS))
    bom = BOM(
        parts=[
            _bpart("U1", ATTINY),
            _bpart("R1", "Device:R"),
            _jpart("J2", "UPDI header 1x03"),
        ],
        connections=[
            NetConnection(
                net_name="UPDI_PULLUP",
                sheet="MCU",
                endpoints=[PinEndpoint(ref="U1", pin="6"), PinEndpoint(ref="R1", pin="2")],
            ),
            NetConnection(
                net_name="+5V",
                sheet="MCU",
                endpoints=[PinEndpoint(ref="U1", pin="1"), PinEndpoint(ref="R1", pin="1")],
            ),
        ],
    )
    res = check_mcu_programming_access(bom)
    assert not res.ok
    assert "UPDI" in res.offenders[0] and "J2" in res.offenders[0]


def test_prog_access_updi_reaching_header_passes(monkeypatch) -> None:
    monkeypatch.setattr(_sp, "lookup_pins", _fake_lookup(ATTINY_PINS))
    bom = BOM(
        parts=[
            _bpart("U1", ATTINY),
            _bpart("R1", "Device:R"),
            _jpart("J2", "UPDI header 1x03"),
        ],
        connections=[
            NetConnection(
                net_name="UPDI",
                sheet="MCU",
                endpoints=[
                    PinEndpoint(ref="U1", pin="6"),
                    PinEndpoint(ref="R1", pin="2"),
                    PinEndpoint(ref="J2", pin="1"),
                ],
            ),
        ],
    )
    assert check_mcu_programming_access(bom).ok


def test_prog_access_non_updi_mcu_not_judged_at_wiring(monkeypatch) -> None:
    # A part whose pinout exposes no UPDI pin (or is unresolvable) is skipped
    # by the reachability half -- SWD-family judgment stays with §9.21.
    monkeypatch.setattr(
        _sp, "lookup_pins", _fake_lookup({"Fake:STM32F103": [("1", "PA13", "bidirectional")]})
    )
    bom = BOM(
        parts=[
            _bpart("U1", "Fake:STM32F103"),
            _jpart("J2", "SWD debug header"),
        ],
        connections=[
            NetConnection(
                net_name="X",
                sheet="MCU",
                endpoints=[PinEndpoint(ref="U1", pin="1"), PinEndpoint(ref="J2", pin="2")],
            ),
        ],
    )
    assert check_mcu_programming_access(bom).ok


def test_a1_usb_dm_rejects_uart_endpoint_without_prescribing_direction(monkeypatch) -> None:
    """Board 783: USB_DM is D-, so RXD0 cannot be moved across R10 to fix a dangle."""
    monkeypatch.setattr(_sp, "lookup_pins", _fake_lookup(_A1_PINS))
    bom = BOM(
        parts=[
            _bpart("R10", "Device:R", sheet="MCU"),
            _bpart(
                "U3",
                "esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1",
                sheet="MCU",
            ),
        ],
        connections=[
            NetConnection(
                net_name="USB_DM",
                sheet="MCU",
                endpoints=[PinEndpoint(ref="R10", pin="1")],
            ),
            NetConnection(
                net_name="USB_DM_MCU",
                sheet="MCU",
                endpoints=[PinEndpoint(ref="R10", pin="2"), PinEndpoint(ref="U3", pin="36")],
            ),
        ],
    )

    offender = check_no_dangling_signal_nets(_a1_arch(), bom).offenders[0]
    _assert_identity_safe(offender)
    assert "required fixed function: IO19" in offender
    assert "RXD0" in offender
    assert "Rejected wrong-function candidate" in offender
    assert "Move the intended load/destination endpoint" not in offender
    assert "do not assume which side is source or destination" in offender
