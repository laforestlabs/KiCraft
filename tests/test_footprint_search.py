"""search_footprints / lookup_footprint: keyword discovery + verification of stock
KiCad footprint ids.

The footprint analog of test_symbol_search. The BOM stage's weak server model guessed
plausible-but-nonexistent stock footprints (e.g. Connector_BarrelJack:BarrelJack_2.1mm_
P5.5mm, which has no .kicad_mod) and, with no footprint-discovery tool, could not find
the real names. These tests pin the discovery + verification behavior. Needs the KiCad
stock footprint libraries, like the symbol tests need the stock symbol libraries.
"""
from __future__ import annotations

import pcbnew
import pytest

import kicraft.design.synthesis.footprint_library as footprint_library
from kicraft.design.synthesis.footprint_library import (
    FootprintNotFoundError,
    lookup_footprint,
    search_footprints,
)


def test_finds_known_pinheader():
    res = search_footprints("pinheader 2x08")
    assert res, "expected 2x08 pin-header footprints"
    assert all("2x08" in s.lower() for s in res)
    assert any("PinHeader_2x08" in s for s in res)


def test_finds_the_exact_p254_vertical_header():
    # The 2x8 2.54mm header the cat-feeder needed; the real id carries the pitch + orientation.
    res = search_footprints("pinheader 2x08 p2.54mm vertical", limit=100)
    assert "Connector_PinHeader_2.54mm:PinHeader_2x08_P2.54mm_Vertical" in res


def test_footprint_stopword_is_dropped():
    # The model habitually appends "footprint"; it must not zero an otherwise-good query.
    assert search_footprints("pinheader 2x08 footprint") == search_footprints("pinheader 2x08")


def test_finds_barreljack_for_the_offender():
    # J1 died on a guessed BarrelJack_2.1mm_P5.5mm; a broad keyword surfaces real ids.
    res = search_footprints("barreljack")
    assert res, "expected barrel-jack footprints"
    assert all("barreljack" in s.lower() for s in res)
    assert any(s.startswith("Connector_BarrelJack:") for s in res)


def test_results_are_library_name_ids():
    res = search_footprints("connector")
    assert res
    assert all(":" in s for s in res)


def test_respects_the_limit():
    assert len(search_footprints("connector", limit=3)) <= 3


def test_empty_for_no_match():
    assert search_footprints("zzqq_nope_xyzzy_nothing") == []


def test_blank_query_returns_empty():
    assert search_footprints("   ") == []


def test_lookup_resolves_pad_count():
    info = lookup_footprint("Resistor_SMD:R_0603_1608Metric")
    assert info["footprint"] == "Resistor_SMD:R_0603_1608Metric"
    assert info["pad_count"] == 2
    assert info["resolved_directory"].endswith("Resistor_SMD.pretty")


def test_lookup_rejects_non_library_name_form():
    with pytest.raises(ValueError):
        lookup_footprint("not_a_colon_form")


def test_lookup_missing_footprint_raises():
    # The exact hallucinated id from the failed cat-feeder run.
    with pytest.raises(FootprintNotFoundError):
        lookup_footprint("Connector_BarrelJack:BarrelJack_2.1mm_P5.5mm")


def test_lookup_rejects_existing_file_that_kicad_cannot_parse(tmp_path, monkeypatch):
    pretty = tmp_path / "Broken.pretty"
    pretty.mkdir()
    (pretty / "Present.kicad_mod").write_text('(footprint "Present" (pad')
    monkeypatch.setattr(
        footprint_library, "resolve_footprint_library_path", lambda *a, **kw: pretty
    )
    with pytest.raises(FootprintNotFoundError):
        lookup_footprint("Broken:Present", project_root=tmp_path)


def test_native_import_preserves_geometry_and_regenerates_all_item_identities(tmp_path):
    import gc

    path = tmp_path / "Part.kicad_mod"
    path.write_text('''(footprint "ImportedName"
      (version 20240108) (generator pcbnew) (layer "F.Cu")
      (uuid "11111111-1111-4111-8111-111111111111")
      (fp_line (start 0 0) (end 2 0)
        (stroke (width 0.1) (type default)) (layer "F.SilkS")
        (uuid "33333333-3333-4333-8333-333333333333"))
      (pad "1" smd rect (at 0 0) (size 1 1)
        (layers "F.Cu" "F.Paste" "F.Mask")
        (uuid "22222222-2222-4222-8222-222222222222"))
    )''')
    first = footprint_library.load_footprint_file(pcbnew, path)
    gc.collect()
    second = footprint_library.load_footprint_file(pcbnew, path)

    def identities(footprint):
        return {
            item.m_Uuid.AsString()
            for item in [footprint, *footprint.Pads(), *footprint.GraphicalItems()]
        }

    assert identities(first).isdisjoint(identities(second))
    assert "11111111-1111-4111-8111-111111111111" not in identities(first)
    assert "22222222-2222-4222-8222-222222222222" not in identities(first)
    assert "33333333-3333-4333-8333-333333333333" not in identities(first)
    board = pcbnew.BOARD()
    first.SetReference("J1")
    second.SetReference("J2")
    second.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(5), 0))
    board.Add(first)
    board.Add(second)
    del first, second
    gc.collect()
    output = tmp_path / "owned.kicad_pcb"
    pcbnew.SaveBoard(str(output), board)
    restored = pcbnew.LoadBoard(str(output))
    geometry = {}
    for footprint in restored.GetFootprints():
        pad = next(iter(footprint.Pads()))
        geometry[footprint.GetReference()] = (
            pad.GetNumber(), pad.GetPosition().x, pad.GetPosition().y,
            pad.GetSize().x, pad.GetSize().y,
        )
    mm = pcbnew.FromMM(1)
    assert geometry == {
        "J1": ("1", 0, 0, mm, mm),
        "J2": ("1", 5 * mm, 0, mm, mm),
    }
