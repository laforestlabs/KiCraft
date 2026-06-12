"""§9.13 netlist faithfulness — the KiCad-extracted netlist must match
bom.connections.

ERC misses two wiring-corruption classes this check catches:
- silent net merges (two labels on one wire is legal KiCad — a de-collision
  pass that slid ISP_MISO onto the ISP_MOSI stub shorted the nets with zero
  ERC errors when the abandoned stub kept another label);
- lost pins (an unescaped quote corrupted a child sheet; KiCad loaded it as
  EMPTY, so its parts vanished from netlist and board while ERC reported only
  an unrelated-looking hier_label_mismatch).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from kicraft.design.models import ConversationState
from kicraft.design.synthesize import run
from kicraft.design.synthesis.validation import (
    _compare_netlist_to_bom,
    _extract_netlist_groups,
    check_netlist_faithfulness,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bmp280_reader_state.json"

needs_kicad = pytest.mark.skipif(
    shutil.which("kicad-cli") is None, reason="kicad-cli not installed"
)


@pytest.fixture
def bmp280_state() -> ConversationState:
    if not _FIXTURE.is_file():
        pytest.skip(f"fixture missing: {_FIXTURE}")
    return ConversationState.model_validate_json(_FIXTURE.read_text())


def test_extract_netlist_groups_parses_nodes() -> None:
    text = (
        '(export (nets\n'
        '  (net (code "1") (name "/SDA")\n'
        '    (node (ref "U1") (pin "3") (pintype "bidirectional"))\n'
        '    (node (ref "R2") (pin "1") (pintype "passive"))\n'
        '    (node (ref "#PWR01") (pin "1")))\n'
        '  (net (code "2") (name "GND")\n'
        '    (node (ref "C1") (pin "2")))))\n'
    )
    groups = _extract_netlist_groups(text)
    assert {("U1", "3"), ("R2", "1")} in groups
    assert {("C1", "2")} in groups
    assert all(("#PWR01", "1") not in g for g in groups)


@needs_kicad
def test_clean_project_passes(tmp_path, bmp280_state) -> None:
    run(bmp280_state, tmp_path)
    res = check_netlist_faithfulness(
        tmp_path, bmp280_state.project_stem, bmp280_state.bom
    )
    assert res.ok, f"{res.message}: {res.offenders}"


def test_compare_flags_cross_net_merge(bmp280_state) -> None:
    """The merge branch, unit-level: pins of two BOM nets sharing neither a
    name nor an endpoint land in one extracted net (what a global/hier label
    on a foreign wire produces) -> reported as a merge."""
    bom = bmp280_state.bom
    two = [c for c in bom.connections if len(c.endpoints) >= 1][:2]
    a, b = two[0], two[1]
    fused = {(a.endpoints[0].ref, str(a.endpoints[0].pin)),
             (b.endpoints[0].ref, str(b.endpoints[0].pin))}
    if a.net_name == b.net_name:
        pytest.skip("fixture's first two connections share a name")
    merges, _lost = _compare_netlist_to_bom([fused], bom)
    assert merges and a.net_name in merges[0] and b.net_name in merges[0]


@needs_kicad
def test_detects_label_slide_artifact(tmp_path, bmp280_state) -> None:
    """Plant the exact artifact the label-slide regression produced — a net
    label relocated from its own wire onto a neighboring net's wire. ERC can
    stay quiet about it (two labels on one wire is legal); KiCad's netlist
    export drops the conflicted nets, so §9.13 must report the wired pins of
    BOTH nets as lost."""
    run(bmp280_state, tmp_path)
    sch = tmp_path / "USB_INPUT.kicad_sch"
    text = sch.read_text()
    labs = list(re.finditer(
        r'\(label "([^"]+)"\s*\(at ([\d.-]+) ([\d.-]+) (\d+)\)', text
    ))
    names = {m.group(1) for m in labs}
    if len(names) < 2:
        pytest.skip("fixture sheet no longer has two distinct local labels")
    lab_a = labs[0]
    lab_b = next(m for m in labs if m.group(1) != lab_a.group(1))
    moved = lab_a.group(0).replace(
        f"(at {lab_a.group(2)} {lab_a.group(3)} {lab_a.group(4)})",
        f"(at {lab_b.group(2)} {lab_b.group(3)} {lab_b.group(4)})",
    )
    sch.write_text(text[: lab_a.start()] + moved + text[lab_a.end():])

    res = check_netlist_faithfulness(
        tmp_path, bmp280_state.project_stem, bmp280_state.bom
    )
    assert not res.ok, "label-slide artifact was not detected"
    assert res.offenders, res.message


@needs_kicad
def test_detects_lost_pins(tmp_path, bmp280_state) -> None:
    """Empty out one child sheet (what KiCad does to an unparseable one) —
    every wired pin on it must be reported lost."""
    run(bmp280_state, tmp_path)
    leaves = [
        p for p in sorted(tmp_path.glob("*.kicad_sch"))
        if p.stem != bmp280_state.project_stem
    ]
    victim = leaves[0]
    text = victim.read_text()
    uuid = re.search(r'\(uuid "([^"]+)"\)', text).group(1)
    victim.write_text(
        "(kicad_sch\n\t(version 20250114)\n"
        '\t(generator "eeschema")\n\t(generator_version "9.0")\n'
        f'\t(uuid "{uuid}")\n\t(paper "A4")\n\t(lib_symbols)\n)\n'
    )

    res = check_netlist_faithfulness(
        tmp_path, bmp280_state.project_stem, bmp280_state.bom
    )
    assert not res.ok, "emptied sheet was not detected"
    assert any("missing from netlist" in o for o in res.offenders), res.offenders
