"""Tests for §9 mechanical validation checks.

Each check has a positive (passing) and negative (failing) case using small
hand-rolled fixture directories. The LLUPS project (live, slightly quirky) is
used as a real-world smoke fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.upstream.synthesis.validation import (
    SynthesisValidationError,
    check_autoplacer_is_valid_json,
    check_footprints_nonempty,
    check_named_refs_exist,
    check_pin_directions,
    check_schematic_version,
    check_sheetfile_refs_resolve,
    run_validations,
)

LLUPS_ROOT = Path("/home/jason/Documents/LLUPS")
LLUPS_AVAILABLE = (LLUPS_ROOT / "LLUPS.kicad_sch").is_file()


# ---------- helpers to build a tiny passing project ----------


def _write_minimal_project(d: Path, stem: str = "TINY") -> None:
    """Two-sheet fixture that passes every §9 check."""
    (d / f"{stem}.kicad_sch").write_text(
        f'(kicad_sch\n'
        f'\t(version 20250114)\n'
        f'\t(generator "eeschema")\n'
        f'\t(uuid "11111111-1111-1111-1111-111111111111")\n'
        f'\t(lib_symbols)\n'
        f'\t(sheet\n'
        f'\t\t(at 30 40) (size 30 15)\n'
        f'\t\t(uuid "22222222-2222-2222-2222-222222222222")\n'
        f'\t\t(property "Sheetname" "REG" (at 30 39 0))\n'
        f'\t\t(property "Sheetfile" "REG.kicad_sch" (at 30 56 0))\n'
        f'\t\t(pin "VBUS" bidirectional (at 60 45 0) (uuid "33333333-3333-3333-3333-333333333333"))\n'
        f'\t)\n'
        f')\n'
    )
    (d / "REG.kicad_sch").write_text(
        '(kicad_sch\n'
        '\t(version 20250114)\n'
        '\t(generator "eeschema")\n'
        '\t(uuid "44444444-4444-4444-4444-444444444444")\n'
        '\t(lib_symbols)\n'
        '\t(symbol\n'
        '\t\t(lib_id "Device:R")\n'
        '\t\t(at 100 80 0)\n'
        '\t\t(uuid "55555555-5555-5555-5555-555555555555")\n'
        '\t\t(property "Reference" "R1" (at 100 70 0))\n'
        '\t\t(property "Value" "10k" (at 100 73 0))\n'
        '\t\t(property "Footprint" "Resistor_SMD:R_0402_1005Metric" (at 100 86 0))\n'
        '\t)\n'
        ')\n'
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
        '(kicad_sch (version 20250114) (lib_symbols)\n'
        '(symbol (lib_id "Device:R") (at 0 0 0)\n'
        '  (property "Reference" "R1" (at 0 0 0))\n'
        '  (property "Footprint" "" (at 0 0 0))\n'
        ')\n)\n'
    )
    r = check_footprints_nonempty(tmp_path)
    assert not r.ok
    assert any("R1" in o for o in r.offenders)


def test_footprints_ignore_lib_symbols_template(tmp_path: Path) -> None:
    # lib_symbols carries an empty Footprint template; placed instance has a real one.
    (tmp_path / "X.kicad_sch").write_text(
        '(kicad_sch (version 20250114)\n'
        '(lib_symbols\n'
        '  (symbol "Device:R"\n'
        '    (property "Reference" "R" (at 0 0 0))\n'
        '    (property "Footprint" "" (at 0 0 0))\n'
        '  )\n'
        ')\n'
        '(symbol (lib_id "Device:R") (at 0 0 0)\n'
        '  (property "Reference" "R1" (at 0 0 0))\n'
        '  (property "Footprint" "Resistor_SMD:R_0402_1005Metric" (at 0 0 0))\n'
        ')\n)\n'
    )
    r = check_footprints_nonempty(tmp_path)
    assert r.ok, r.offenders


def test_footprints_ignore_power_symbols(tmp_path: Path) -> None:
    # #PWR is KiCad's power-flag pseudo-component; its empty Footprint is correct.
    (tmp_path / "X.kicad_sch").write_text(
        '(kicad_sch (version 20250114) (lib_symbols)\n'
        '(symbol (lib_id "power:GND") (at 0 0 0)\n'
        '  (property "Reference" "#PWR0042" (at 0 0 0))\n'
        '  (property "Footprint" "" (at 0 0 0))\n'
        ')\n)\n'
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
        '(kicad_sch (version 20250114) (lib_symbols)\n'
        '(sheet (at 0 0) (size 10 5)\n'
        '  (pin "VBUS" power_in (at 0 0 0) (uuid "x"))\n'
        ')\n)\n'
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
        '(kicad_sch (version 20250114) (lib_symbols)\n'
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
