"""Researching a demanded part class into a real, recorded part.

Hermetic: the catalog is a tiny sqlite fixture, the fetch and the symbol reader are injected, and
the record file is a scratch path -- no network, no 5 GB dump, no KiCad libraries. What is being
pinned is the behaviour the pipeline relies on: a class no reviewed part carries gets a real part
searched out of the catalog, chosen on evidence, vendored, recorded **once**, and covered from then
on -- in this process and in every later one.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from kicraft.design import part_identity, part_research
from kicraft.parts_library import researched_records

_SCHEMA = """CREATE TABLE jlc_components (
    lcsc INTEGER PRIMARY KEY NOT NULL, mfr TEXT NOT NULL, package TEXT NOT NULL,
    manufacturer TEXT NOT NULL, library_type TEXT NOT NULL, stock INTEGER NOT NULL,
    price TEXT NOT NULL, description TEXT NOT NULL, joints INTEGER)"""

# One good part, one dry one, one dual-device array, and one whose class word sits past the
# catalog's own description truncation.
_ROWS = [
    (1, "TFM252012ALMA100MTAA", "0603", "TDK", "expand", 120000, "1-:0.08",
     "10uH 1.2A 0603 Inductors ROHS"),
    (2, "DRY-PART", "0603", "NoStock", "expand", 5, "1-:0.05",
     "10uH 0603 Inductors ROHS"),
    (3, "DUAL-CHOKE", "0805", "Array", "expand", 90000, "1-:0.2",
     "2x10uH Common Mode Inductors ROHS"),
]

_MANIFEST = {
    "name": "inductor-c1",
    "mpn": "TFM252012ALMA100MTAA",
    "symbol_name": "L_TFM252012",
    "footprint_name": "L_0603",
    "description": "10uH 1.2A power inductor, 0603",
    "sourcing": {"lcsc": "C1"},
    "datasheet_url": "https://example.invalid/tdk.pdf",
}


@pytest.fixture(autouse=True)
def _fresh_research_memo():
    """The loop memoises per class per process; each test starts from no research."""
    part_research._RESEARCHED_THIS_PROCESS.clear()
    yield
    part_research._RESEARCHED_THIS_PROCESS.clear()


@pytest.fixture
def catalog(tmp_path, monkeypatch) -> Path:
    db = tmp_path / "cache.sqlite3"
    con = sqlite3.connect(db)
    con.execute(_SCHEMA)
    con.executemany(
        "INSERT INTO jlc_components (lcsc, mfr, package, manufacturer, library_type, stock, "
        "price, description) VALUES (?,?,?,?,?,?,?,?)",
        _ROWS,
    )
    con.commit()
    con.close()
    monkeypatch.setenv("KICRAFT_JLCPARTS_DB", str(db))
    return db


@pytest.fixture
def record_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "researched_records.json"
    monkeypatch.setenv("KICRAFT_RESEARCHED_RECORDS", str(path))
    return path


def _fetcher(calls: list[dict]):
    def fetch(lcsc, slug, **kwargs):
        calls.append({"lcsc": lcsc, "slug": slug, **kwargs})
        return {**_MANIFEST, "name": slug, "sourcing": {"lcsc": lcsc}}

    return fetch


def test_research_finds_vendors_and_records_a_part(catalog, record_file) -> None:
    calls: list[dict] = []
    result = part_research.research_part_for_class(
        "inductor", fetcher=_fetcher(calls), pins_reader=lambda symbol: ("1", "2")
    )

    assert result is not None
    assert (result.identity, result.lcsc, result.fetched) == ("tfm252012alma100mtaa", "C1", True)
    assert "inductor" in result.reason and "120000" in result.reason
    assert calls == [
        {"lcsc": "C1", "slug": "inductor-c1", "component_class": "inductor",
         "description": "10uH 1.2A 0603 Inductors ROHS"}
    ]

    stored = json.loads(record_file.read_text())[0]
    assert stored["family"] == "inductor"
    assert stored["symbol"] == "inductor-c1:L_TFM252012"
    assert stored["footprint"] == "inductor-c1:L_0603"
    assert stored["contacts"] == ["1", "2"]
    assert stored["physical_features"] == ["inductor"]
    assert stored["researched"]["from"].startswith("jlcparts")
    # No rating is claimed: nothing read a datasheet, and an absent limit means unproven.
    assert "operating_limits" not in stored or stored["operating_limits"] == {}

    # The class is covered from here on -- for this process and, because the record is on disk,
    # for the next one too.
    carriers = part_identity.reviewed_parts_for_feature("inductor")
    assert [part.identity for part in carriers] == ["tfm252012alma100mtaa"]
    assert part_identity.reviewed_part("TFM252012ALMA100MTAA") is not None
    assert part_identity.has_reviewed_coverage("inductor")

    # A second call does nothing: the class is covered, so no part is fetched again.
    again = part_research.research_part_for_class("inductor", fetcher=_fetcher(calls))
    assert again is None and len(calls) == 1


def test_the_dry_and_dual_rows_are_rejected(catalog, record_file) -> None:
    chosen, reason = part_research.choose_candidate("inductor", part_research.search_candidates("inductor"))
    assert chosen is not None and chosen["lcsc"] == "C1"
    assert "in stock (120000)" in reason

    # Only the dry part available: nothing qualifies rather than a part nobody can buy.
    rows = [row for row in part_research.search_candidates("inductor") if row["lcsc"] == "C2"]
    chosen, reason = part_research.choose_candidate("inductor", rows)
    assert chosen is None and "out of stock" in reason


def test_a_catalogue_row_whose_class_word_is_truncated_still_counts(catalog, record_file) -> None:
    """The catalog hands back a truncated description; the phrase that found the row is evidence."""
    row = {"lcsc": "C9", "model": "X1", "package": "0603", "stock": 5000, "price": 0.01,
           "description": "", "_matched_phrase": "inductor"}
    chosen, _reason = part_research.choose_candidate("inductor", [row])
    assert chosen is not None


def test_a_failed_fetch_records_nothing(catalog, record_file) -> None:
    result = part_research.research_part_for_class(
        "inductor", fetcher=lambda *args, **kwargs: None
    )
    assert result is None
    assert not record_file.exists()
    assert part_identity.reviewed_parts_for_feature("inductor") == ()


def test_a_package_requirement_that_no_candidate_meets_records_nothing(catalog, record_file) -> None:
    result = part_research.research_part_for_class(
        "inductor",
        package="SOT-23",
        fetcher=_fetcher([]),
        pins_reader=lambda symbol: (),
    )
    assert result is None
    assert not record_file.exists()


def test_the_batch_cap_bounds_one_pass(catalog, record_file) -> None:
    calls: list[dict] = []
    results = part_research.research_uncovered_classes(
        ["inductor", "varistor", "ferrite-bead"],
        limit=1,
        fetcher=_fetcher(calls),
        pins_reader=lambda symbol: ("1", "2"),
    )
    assert [result.component_class for result in results] == ["inductor"]
    assert len(calls) == 1


def test_the_stage_completion_answers_a_demanded_class_before_diagnosis(monkeypatch, record_file) -> None:
    """The pipeline hook: a demand no reviewed part answers is researched, and said so plainly."""
    from kicraft.design.stage_semantics import complete_unavailable_part_classes

    result = part_research.ResearchResult(
        component_class="varistor",
        identity="10d471k",
        lcsc="C8760",
        symbol="varistor-c8760:10D471K-C8760",
        footprint="varistor-c8760:RES-TH_L12.5-W6.7-P7.50-D1.2",
        fetched=True,
        reason="picked from the catalog",
    )
    monkeypatch.setattr(part_research, "research_part_for_class", lambda *a, **k: result)

    completed = complete_unavailable_part_classes(
        "architecture",
        {"assumptions": ["something already noted (defaulted)"]},
        {"intent": {"obligations": [{"kind": "physical", "component_class": "varistor"}]}},
    )

    notes = completed["assumptions"]
    assert notes[0] == "something already noted (defaulted)"
    assert "varistor" in notes[1] and "10d471k" in notes[1] and "defaulted" in notes[1]

    # A class that is already answered is left alone, and so is a stage that does not own parts.
    assert complete_unavailable_part_classes(
        "architecture", {"assumptions": []}, {"intent": {"obligations": [
            {"kind": "physical", "component_class": "led-0603"}]}}
    ) == {"assumptions": []}
    assert complete_unavailable_part_classes(
        "intent", {"assumptions": []}, {"intent": {"obligations": [
            {"kind": "physical", "component_class": "varistor"}]}}
    ) == {"assumptions": []}


def test_records_round_trip_and_replace_by_part(record_file, tmp_path) -> None:
    first = {"identity": "abc123", "family": "varistor", "lcsc": "C1", "contacts": ["2", "1"],
             "physical_features": ["varistor"]}
    researched_records.append(first, record_file)
    assert researched_records.load(record_file)[0]["contacts"] == ["1", "2"]

    # Re-researching the same part replaces its record rather than stacking a second one.
    better = {**first, "package": "SOD-123", "lcsc": "C1"}
    researched_records.append(better, record_file)
    stored = researched_records.load(record_file)
    assert len(stored) == 1 and stored[0]["package"] == "SOD-123"

    # A hand-edited file that is not JSON must not take the pipeline down with it.
    record_file.write_text("{not json")
    assert researched_records.load(record_file) == ()


def test_coverage_reads_the_record_file(record_file) -> None:
    researched_records.append(
        {"identity": "x1", "family": "varistor", "physical_features": ["varistor", "tvs"]},
        record_file,
    )
    assert researched_records.coverage(record_file) == {
        "tvs": ("x1",),
        "varistor": ("x1",),
    }
