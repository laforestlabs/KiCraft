"""Offline JLC catalog (jlcparts dump): reads, split-zip extraction, updater.

Everything runs against tiny fixture databases/archives in tmp_path — no
network, no real 5 GB catalog. The updater is exercised end-to-end over
file:// URLs.
"""
from __future__ import annotations

import io
import sqlite3
import zipfile
from pathlib import Path

import pytest

from kicraft.parts_library import jlcparts

_SCHEMA = """CREATE TABLE jlc_components (
    lcsc INTEGER PRIMARY KEY NOT NULL, mfr TEXT NOT NULL, package TEXT NOT NULL,
    manufacturer TEXT NOT NULL, library_type TEXT NOT NULL, stock INTEGER NOT NULL,
    price TEXT NOT NULL, description TEXT NOT NULL)"""

_ROWS = [
    # lcsc, mfr, package, manufacturer, type, stock, price, description
    (190004, "VL53L1CXV0FY/1", "LGA-12(2.5x4.9)", "STMicroelectronics", "expand",
     5640, "1-9:4.817,10-29:4.2745,100-499:3.3492,1000-:3.1586",
     "4m I2C  LGA-12(2.5x4.9) Position Sensors ROHS"),
    (2924337, "VL53L1X", "LGA-12", "", "expand",
     0, "1-999:0.0007,1000-:0.0003", "LGA-12 New Arrivals ROHS"),
    (25744, "RC0805FR-07100KL", "0805", "YAGEO", "base",
     400000, "1-:0.0041", "100kOhm +-1% 0805 Chip Resistor ROHS"),
    (1525, "CL21B104KBCNNNC", "-", "Samsung", "base",
     900000, "1-:0.0046", "50V 100nF X7R +-10% 0805 MLCC ROHS"),
]


@pytest.fixture
def catalog(tmp_path, monkeypatch) -> Path:
    db = tmp_path / "cache.sqlite3"
    con = sqlite3.connect(db)
    con.execute(_SCHEMA)
    con.executemany("INSERT INTO jlc_components VALUES (?,?,?,?,?,?,?,?)", _ROWS)
    con.commit()
    con.close()
    monkeypatch.setenv("KICRAFT_JLCPARTS_DB", str(db))
    return db


# ------------------------------------------------------------------- reads

def test_unavailable_when_db_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_JLCPARTS_DB", str(tmp_path / "nope.sqlite3"))
    assert jlcparts.available() is False
    assert jlcparts.search("VL53L1X") == []
    assert jlcparts.lookup("C190004") is None


def test_search_exact_mpn_and_candidate_shape(catalog):
    rows = jlcparts.search("vl53l1cxv0fy/1")          # case-insensitive exact
    assert [r["lcsc"] for r in rows] == ["C190004"]
    r = rows[0]
    assert r["model"] == "VL53L1CXV0FY/1"
    assert r["brand"] == "STMicroelectronics"
    assert r["type"] == "Extended"
    assert r["stock"] == 5640
    assert r["price"] == 4.817                          # qty-1 ladder price


def test_search_substring_orders_in_stock_first(catalog):
    rows = jlcparts.search("VL53L1")
    assert [r["lcsc"] for r in rows] == ["C190004", "C2924337"]  # stock desc


def test_search_terms_fallback_over_description(catalog):
    rows = jlcparts.search("100nF 0805 X7R")
    assert [r["lcsc"] for r in rows] == ["C1525"]
    assert rows[0]["type"] == "Basic"
    assert rows[0]["package"] is None                   # '-' normalized to None


def test_parse_ladder_and_price_at():
    ladder = jlcparts.parse_ladder("1-9:4.817,10-29:4.27,garbage,1000-:3.16")
    assert ladder == [{"qty_from": 1, "qty_to": 9, "price": 4.817},
                      {"qty_from": 10, "qty_to": 29, "price": 4.27},
                      {"qty_from": 1000, "qty_to": None, "price": 3.16}]
    assert jlcparts.price_at(ladder, 1) == 4.817
    assert jlcparts.price_at(ladder, 10) == 4.27
    assert jlcparts.price_at(ladder, 5000) == 3.16
    assert jlcparts.price_at(ladder, 50) is None        # gap in this ladder


def test_lookup_by_id_with_ladder(catalog):
    part = jlcparts.lookup("C190004")
    assert part["model"] == "VL53L1CXV0FY/1"
    assert part["ladder"][0]["price"] == 4.817
    assert jlcparts.lookup(190004)["lcsc"] == "C190004"
    assert jlcparts.lookup("C424242") is None
    assert jlcparts.lookup("not-an-id") is None


# ------------------------------------------------- split-zip extraction

def _make_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def _split(blob: bytes, tmp_path: Path, cuts: list[int]) -> list[Path]:
    """Chop blob at byte offsets into cache.z01..zNN + cache.zip (last)."""
    bounds = [0, *cuts, len(blob)]
    parts = [blob[a:b] for a, b in zip(bounds, bounds[1:])]
    paths = []
    for i, part in enumerate(parts):
        name = "cache.zip" if i == len(parts) - 1 else f"cache.z{i + 1:02d}"
        p = tmp_path / name
        p.write_bytes(part)
        paths.append(p)
    return paths


def test_extract_split_zip_multi_volume(tmp_path):
    members = {"cache.sqlite3": bytes(range(256)) * 4000, "note.txt": b"hi"}
    blob = _make_zip(members)
    vols = _split(blob, tmp_path, [len(blob) // 3, 2 * len(blob) // 3])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    outs = jlcparts.extract_split_zip(vols, out_dir)
    assert {p.name for p in outs} == set(members)
    for p in outs:
        assert p.read_bytes() == members[p.name]


def test_extract_single_volume(tmp_path):
    blob = _make_zip({"cache.sqlite3": b"x" * 10_000})
    p = tmp_path / "cache.zip"
    p.write_bytes(blob)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out,) = jlcparts.extract_split_zip([p], out_dir)
    assert out.read_bytes() == b"x" * 10_000


def test_extract_rejects_non_zip(tmp_path):
    p = tmp_path / "cache.zip"
    p.write_bytes(b"definitely not a zip" * 10)
    with pytest.raises(ValueError):
        jlcparts.extract_split_zip([p], tmp_path)


# ------------------------------------------------------------- updater

def test_update_end_to_end_over_file_urls_prunes_low_stock(tmp_path):
    # A plausible catalog: big enough to pass the row-count sanity check.
    # Even rows are in stock (7), odd rows are low stock (1) -> pruned.
    src_db = tmp_path / "src.sqlite3"
    con = sqlite3.connect(src_db)
    con.execute(_SCHEMA)
    con.executemany(
        "INSERT INTO jlc_components VALUES (?,?,?,?,?,?,?,?)",
        ((i, f"P{i}", "0805", "m", "base", 7 if i % 2 == 0 else 1, "1-:0.01", "r")
         for i in range(100_002)))
    con.execute("CREATE TABLE lcsc_components (lcsc INTEGER PRIMARY KEY)")
    con.commit()
    con.close()

    site = tmp_path / "site"
    site.mkdir()
    blob = _make_zip({"cache.sqlite3": src_db.read_bytes()})
    _split(blob, site, [len(blob) // 2])               # cache.z01 + cache.zip

    dest = tmp_path / "installed" / "cache.sqlite3"
    msgs = []
    stats = jlcparts.update(dest=dest, base_url=site.as_uri() + "/",
                            progress=msgs.append)
    assert stats["rows"] == 50_001 and stats["pruned"] == 50_001
    con = sqlite3.connect(dest)
    assert con.execute("SELECT COUNT(*) FROM jlc_components").fetchone()[0] == 50_001
    assert con.execute("SELECT MIN(stock) FROM jlc_components").fetchone()[0] >= 5
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
    con.close()
    assert "idx_jlc_mfr" in names                       # exact-MPN lookups stay fast
    assert "lcsc_components" not in names               # unused side table dropped
    assert not (dest.parent / "update.tmp").exists()    # downloads cleaned up
    assert any("installed" in m for m in msgs)


def test_search_prefix_fallback_finds_family_in_pruned_catalog(tmp_path, monkeypatch):
    # A pruned catalog has no zero-stock placeholder row for "VL53L1X" at all;
    # the bare-family query must still surface the orderable MPN via the
    # bounded prefix fallback.
    db = tmp_path / "pruned.sqlite3"
    con = sqlite3.connect(db)
    con.execute(_SCHEMA)
    con.execute("INSERT INTO jlc_components VALUES (?,?,?,?,?,?,?,?)",
                (190004, "VL53L1CXV0FY/1", "LGA-12", "ST", "expand", 5640,
                 "1-:4.8", "ToF"))
    con.commit()
    con.close()
    monkeypatch.setenv("KICRAFT_JLCPARTS_DB", str(db))
    rows = jlcparts.search("VL53L1X")
    assert [r["lcsc"] for r in rows] == ["C190004"]


def test_update_failure_leaves_existing_catalog(tmp_path):
    dest = tmp_path / "cache.sqlite3"
    dest.write_bytes(b"previous catalog")
    empty = tmp_path / "empty-site"
    empty.mkdir()
    with pytest.raises(Exception):
        jlcparts.update(dest=dest, base_url=empty.as_uri() + "/")
    assert dest.read_bytes() == b"previous catalog"


def test_search_widens_zero_stock_exact_hit_to_family(catalog):
    # "VL53L1X" exactly matches only the zero-stock placeholder; the widened
    # result set must surface the in-stock real MPN first.
    rows = jlcparts.search("VL53L1X")
    assert rows[0]["lcsc"] == "C190004" and rows[0]["stock"] == 5640
    assert {r["lcsc"] for r in rows} >= {"C190004", "C2924337"}
