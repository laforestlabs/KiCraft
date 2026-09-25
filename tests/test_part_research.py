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
    price TEXT NOT NULL, description TEXT NOT NULL, joints INTEGER,
    attributes TEXT NOT NULL DEFAULT '{}', datasheet TEXT NOT NULL DEFAULT '')"""

# The chosen part's own parametric table, as the catalog publishes it: real field names, real
# values, including two the loop claims no rating from.
_ATTRIBUTES = (
    '{"Inductance": "10uH", "Current Rating": "1.2A", "DC Resistance(DCR)": "100mΩ",'
    ' "Operating Temperature": "-40℃~+85℃", "Type": "Unshielded Inductor"}'
)

# One good part, one dry one, one dual-device array, and one whose class word sits past the
# catalog's own description truncation.
_ROWS = [
    (1, "TFM252012ALMA100MTAA", "0603", "TDK", "expand", 120000, "1-:0.08",
     "10uH 1.2A 0603 Inductors ROHS", _ATTRIBUTES, "https://example.invalid/C1.pdf"),
    (2, "DRY-PART", "0603", "NoStock", "expand", 5, "1-:0.05",
     "10uH 0603 Inductors ROHS", "{}", ""),
    (3, "DUAL-CHOKE", "0805", "Array", "expand", 90000, "1-:0.2",
     "2x10uH Common Mode Inductors ROHS", "{}", ""),
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
        "price, description, attributes, datasheet) VALUES (?,?,?,?,?,?,?,?,?,?)",
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
    assert result.limits["inductance_h"] == 1e-05
    assert "inductance_h 1e-05" in result.ratings_note()
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

    # The ratings are the catalog's own parametric fields, converted and nothing more: 10 uH is
    # 1e-05 H, 100 mOhm is 0.1 Ohm, and the temperature range becomes the pair the pipeline reads.
    assert stored["operating_limits"] == {
        "inductance_h": 1e-05,
        "current_a": 1.2,
        "dc_resistance_ohm": 0.1,
        "temperature_min_c": -40,
        "temperature_max_c": 85,
    }
    assert stored["limits_source"]["inductance_h"] == "catalog parameter 'Inductance' = '10uH'"
    assert stored["limits_review"]["status"] == "catalog-parameters"
    # No rating is read off a field that does not state one -- and it is kept, not dropped.
    assert stored["unrated_parameters"] == {"Type": "Unshielded Inductor"}
    # The catalog's own datasheet link is cited beside the vendored one.
    assert "https://example.invalid/C1.pdf" in stored["manufacturer_sources"]

    # The class is covered from here on -- for this process and, because the record is on disk,
    # for the next one too.
    carriers = part_identity.reviewed_parts_for_feature("inductor")
    assert [part.identity for part in carriers] == ["tfm252012alma100mtaa"]
    assert part_identity.reviewed_part("TFM252012ALMA100MTAA") is not None
    assert part_identity.has_reviewed_coverage("inductor")
    # ...and a check that reads a record's ratings sees the researched part's, which is the
    # difference between "exists and is in stock" and "is right for this rail".
    assert carriers[0].operating_limits["inductance_h"] == 1e-05
    assert part_identity.reviewed_supply_voltage_limits(carriers[0]) == []

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


def test_the_result_states_the_ratings_it_read_in_one_line() -> None:
    """The hand-run command shows what the record claims, so the owner never opens the JSON."""
    rated = part_research.ResearchResult(
        component_class="schottky-diode", identity="b340a", lcsc="C64982", symbol="s", footprint="f",
        fetched=True, reason="picked", limits={"reverse_voltage_v": 40.0, "rectified_current_a": 3.0},
    )
    assert rated.ratings_note() == "rectified_current_a 3, reverse_voltage_v 40"

    unrated = part_research.ResearchResult(
        component_class="varistor", identity="x", lcsc="C9", symbol="s", footprint="f",
        fetched=True, reason="picked",
    )
    assert "every limit is unverified" in unrated.ratings_note()


def test_a_researched_parts_supply_range_refuses_a_rail_outside_it(record_file) -> None:
    """The point of reading the catalog's ratings: a rail outside the part's own range is refused.

    Without them the part was merely "in stock", and the stage had nothing to compare the rail
    against; with them the same rail is a stated fault naming the rating it broke.
    """
    from kicraft.design import stage_semantics

    researched_records.append(
        part_research.build_record(
            "linear-regulator",
            {"lcsc": "C1", "description": "a 3.3 V linear regulator"},
            {"name": "ldo-c1", "mpn": "XC6206", "sourcing": {"lcsc": "C1"}},
            attributes={"Voltage - Supply": "2.6V~6V"},
        ),
        record_file,
    )

    def candidate(volts: float) -> dict:
        return {
            "requirements": [{
                "id": "ldo", "family": "linear-regulator", "role": "regulator",
                "exact_part": "xc6206", "supply": "VIN",
                "ports": {"input": "VIN", "output": "+3V3"},
            }],
            "power": {"rails": {"VIN": {"voltage": volts}, "+3V3": {"voltage": 3.3}}},
        }

    refused = stage_semantics._architecture_supply_over_rating(candidate(9.0))
    assert [diagnostic.code for diagnostic in refused] == [
        "architecture_supply_exceeds_part_rating"
    ]
    assert stage_semantics._architecture_supply_over_rating(candidate(5.0)) == []


def test_catalog_ratings_convert_the_published_units() -> None:
    """Real catalog fields and values, on the parts a brief actually demands."""
    ratings, sources, unrated = part_research.catalog_ratings({
        "Voltage - DC Reverse(Vr)": "40V",
        "Current - Rectified": "3A",
        "Voltage - Forward(Vf@If)": "550mV@3A",
        "RDS(on)": "12mΩ@10V",
        "Capacitance": "190pF@1kHz",
        "Voltage - Supply": "2V~6V",
        "Number of Pins": "3P",
        "Pitch": "2.54mm",
        "Operating Junction Temperature Range": "-55℃~+150℃",
    })

    assert ratings["reverse_voltage_v"] == 40.0
    assert ratings["rectified_current_a"] == 3.0
    assert ratings["forward_voltage_v"] == 0.55          # the 3 A test point is not the rating
    assert ratings["rds_on_ohm"] == 0.012
    assert ratings["capacitance_f"] == 1.9e-10
    assert (ratings["supply_min_v"], ratings["supply_max_v"]) == (2.0, 6.0)
    assert ratings["positions"] == 3.0
    assert ratings["pitch_mm"] == 2.54
    assert (ratings["temperature_min_c"], ratings["temperature_max_c"]) == (-55.0, 150.0)
    assert sources["reverse_voltage_v"] == "catalog parameter 'Voltage - DC Reverse(Vr)' = '40V'"
    assert unrated == {}


def test_a_field_that_does_not_state_a_rating_is_kept_not_claimed() -> None:
    """A leakage at a voltage, a tolerance, a dielectric: real facts, but no limit to claim."""
    ratings, _sources, unrated = part_research.catalog_ratings({
        "Reverse Leakage Current (Ir)": "500uA@40V",
        "Tolerance": "±5%",
        "Temperature Coefficient": "±100ppm/℃",
        "Voltage - Supply": "3.3V",                      # a nominal, not a range
        "Varistor Voltage": "423V~517V",                 # a range no limit name carries
    })

    assert ratings == {}
    assert unrated["Voltage - Supply"] == "3.3V"
    assert unrated["Reverse Leakage Current (Ir)"] == "500uA@40V"
    assert "Varistor Voltage" in unrated and "Tolerance" in unrated


def test_a_rating_keeps_the_unit_its_own_records_use() -> None:
    """A milliamp is stored the way the reviewed LED records state it, not in amperes."""
    ratings, _sources, _unrated = part_research.catalog_ratings({
        "Forward Current": "20mA",
        "Current Consumption": "10mA",
    })

    assert ratings == {"forward_current_ma": 20.0, "current_consumption_ma": 10.0}


def test_a_range_end_that_drops_its_unit_takes_the_other_ends() -> None:
    """The catalog writes `"-55~85℃"` as often as `"-55℃~+85℃"`; both are one quantity."""
    ratings, _sources, _unrated = part_research.catalog_ratings({
        "Operating Temperature": "-55~85℃",
        "Voltage - Supply": "2~6",
    })

    assert (ratings["temperature_min_c"], ratings["temperature_max_c"]) == (-55.0, 85.0)
    # Two bare numbers are not a range in any unit, so no supply is claimed from them.
    assert "supply_max_v" not in ratings


def test_a_record_with_no_ratings_says_unverified_rather_than_leaving_it_blank() -> None:
    record = part_research.build_record(
        "varistor",
        {"lcsc": "C9", "description": "a varistor"},
        {"name": "varistor-c9", "sourcing": {"lcsc": "C9"}},
        attributes={},
    )

    assert record["operating_limits"] == {}
    assert record["limits_review"]["status"] == "unverified"
    assert "unverified rather than absent" in record["limits_review"]["reason"]


def test_a_researched_rating_reaches_the_checks_that_read_a_record() -> None:
    """The point of the whole thing: a gate sees the researched part's own numbers."""
    record = part_research.build_record(
        "schottky-diode",
        {"lcsc": "C64982", "description": "a Schottky"},
        {"name": "schottky-c64982", "mpn": "B340A", "sourcing": {"lcsc": "C64982"}},
        attributes={"Voltage - DC Reverse(Vr)": "40V", "Current - Rectified": "3A",
                    "Voltage - Supply": "2V~6V"},
    )
    part = part_identity._reviewed_part_from_record(record)

    assert part.operating_limits["reverse_voltage_v"] == 40.0
    assert part.operating_limits["rectified_current_a"] == 3.0
    # A published supply range is a pair, exactly as a hand-reviewed regulator record carries one,
    # so the rail this part is powered from is compared against a real range.
    assert part_identity.reviewed_supply_voltage_limits(
        {"operating_limits": dict(part.operating_limits)}
    ) == [("supply", 2.0, 6.0)]
