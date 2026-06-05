"""add-part sanitizes characters illegal in a KiCad 'Library:Name'.

EasyEDA/LCSC symbol names routinely carry a '#' (e.g. 'DS3231SN#_C722469').
Left unsanitized it flows into the manifest + the on-disk (symbol "...") header,
so the BOM reference 'lib:DS3231SN#_C722469' fails BomPart's SYMBOL_RE and the
fetched part is unusable. The --from-lcsc path needs network; this exercises the
identical sanitize/normalize logic through the hermetic --symbol/--footprint path.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from kicraft.design.cli_app import _sanitize_kicad_name
from kicraft.design.models import SYMBOL_RE
from kicraft.parts_library import load_manifest, project_parts_dir


def _run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "kicraft.design.cli_app", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


# A symbol whose name carries the EasyEDA '#'; the sub-symbol unit mirrors it.
_HASH_SYMBOL = (
    "(kicad_symbol_lib\n"
    '\t(version 20231120)\n'
    '\t(generator "external_tool")\n'
    '\t(symbol "DS3231SN#_C722469"\n'
    '\t\t(property "Reference" "U" (at 0 0 0))\n'
    '\t\t(symbol "DS3231SN#_C722469_1_1"\n'
    '\t\t\t(pin input line (at 0 0 0) (length 2.54)\n'
    '\t\t\t\t(name "VCC") (number "1")\n'
    '\t\t\t)\n'
    '\t\t)\n'
    '\t)\n'
    ")\n"
)

_MIN_FOOTPRINT = (
    '(footprint "SOIC-16"\n'
    "\t(version 20231120)\n"
    '\t(generator "external_tool")\n'
    '\t(layer "F.Cu")\n'
    ")\n"
)


def test_sanitize_kicad_name_strips_illegal_keeps_legal():
    assert _sanitize_kicad_name("DS3231SN#_C722469") == "DS3231SN_C722469"
    # the full legal set is preserved
    assert _sanitize_kicad_name("R_0603_1608Metric.+-") == "R_0603_1608Metric.+-"
    # spaces, slashes, parens, quotes, '#' all stripped
    assert _sanitize_kicad_name("a b/c(d)'e#") == "abcde"
    # the whole point: the sanitized name yields a legal 'Library:Name'
    assert SYMBOL_RE.match(f"ds3231:{_sanitize_kicad_name('DS3231SN#_C722469')}")


def test_add_part_sanitizes_hash_in_symbol_name(
    isolated_home, clean_extras_env, tmp_path
):
    sym = tmp_path / "ds3231.kicad_sym"
    sym.write_text(_HASH_SYMBOL)
    fp = tmp_path / "ds3231.kicad_mod"
    fp.write_text(_MIN_FOOTPRINT)

    project = tmp_path / "project"
    project.mkdir()
    res = _run_cli(
        "add-part",
        "--symbol", str(sym),
        "--footprint", str(fp),
        "--mpn", "DS3231SN",
        "--name", "ds3231",
        cwd=project,
    )
    assert res.returncode == 0, res.stderr

    part_dir = project_parts_dir(project) / "ds3231"
    manifest = load_manifest(part_dir)

    # The reference recorded in the manifest is now legal and '#'-free.
    assert manifest.symbol_name == "DS3231SN_C722469"
    assert SYMBOL_RE.match(f"{manifest.name}:{manifest.symbol_name}")

    # The on-disk symbol header was rewritten to match (no '#' anywhere).
    sym_text = (part_dir / "ds3231.kicad_sym").read_text()
    assert '(symbol "DS3231SN_C722469"' in sym_text
    assert "#" not in sym_text
