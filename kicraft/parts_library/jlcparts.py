"""Offline JLCPCB/LCSC parts catalog (the yaqwsx/jlcparts nightly dump).

jlcpcb.com's anonymous parts API is WAF-blocked from server IPs, so KiCraft
reads the same data from the jlcparts project's published SQLite dump instead:
7M+ LCSC parts with MPN, package, manufacturer, Basic/Extended tier, live
stock, and full quantity-ladder pricing — all answered offline.

``kicraft jlcparts-update`` downloads the dump (a split zip on the project's
GitHub Pages, ~650 MB compressed -> ~5.5 GB sqlite) and atomically swaps it
into place; readers open the DB per call, so a swap never breaks a running
server. When the DB is absent every reader degrades to "no data" and callers
fall through to their network paths.
"""
from __future__ import annotations

import os
import re
import shutil
import sqlite3
import struct
import urllib.request
import zlib
from pathlib import Path

DATA_URL = "https://yaqwsx.github.io/jlcparts/data/"

_CANDIDATE_COLS = ("lcsc, mfr, package, manufacturer, library_type, stock, "
                   "price, description")


# --------------------------------------------------------------------- reads

def db_path() -> Path:
    env = os.environ.get("KICRAFT_JLCPARTS_DB")
    return Path(env) if env else Path.home() / ".kicraft" / "jlcparts" / "cache.sqlite3"


def available() -> bool:
    try:
        return db_path().stat().st_size > 0
    except OSError:
        return False


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def parse_ladder(price: str) -> list[dict]:
    """'1-9:4.817,10-29:4.27,1000-:3.16' -> [{'qty_from','qty_to','price'}, ...]
    (qty_to None = no upper bound). Malformed segments are skipped."""
    out = []
    for seg in (price or "").split(","):
        rng, _, val = seg.partition(":")
        lo, _, hi = rng.partition("-")
        try:
            out.append({"qty_from": int(lo), "qty_to": int(hi) if hi else None,
                        "price": float(val)})
        except ValueError:
            continue
    return out


def price_at(ladder: list[dict], qty: int) -> float | None:
    for step in ladder:
        if step["qty_from"] <= qty and (step["qty_to"] is None or qty <= step["qty_to"]):
            return step["price"]
    return None


def _candidate(row: sqlite3.Row) -> dict:
    ladder = parse_ladder(row["price"])
    desc = re.sub(r"\s+", " ", row["description"] or "").strip()
    return {
        "lcsc": f"C{row['lcsc']}",
        "model": row["mfr"] or None,
        "brand": row["manufacturer"] or None,
        "package": (row["package"] or None) if row["package"] != "-" else None,
        "stock": row["stock"],
        "type": "Basic" if row["library_type"] == "base" else "Extended",
        "price": price_at(ladder, 1),
        "description": desc[:120] or None,
    }


def search(query: str, limit: int = 10) -> list[dict]:
    """Offline candidate search: exact MPN, then MPN substring, then all
    query terms ANDed over mfr+description+manufacturer+package. Rows are
    ordered in-stock-first so callers' exact-match/stock ranking works."""
    q = (query or "").strip()
    if not q or not available():
        return []
    order = "ORDER BY stock DESC, library_type ASC LIMIT ?"
    sel = f"SELECT {_CANDIDATE_COLS} FROM jlc_components"
    con = _connect()
    try:
        rows = con.execute(f"{sel} WHERE mfr = ? COLLATE NOCASE {order}",
                           (q, limit)).fetchall()
        if rows and all(r["stock"] == 0 for r in rows):
            # An exact hit that is entirely out of stock is often a placeholder
            # row for a bare family name (e.g. "VL53L1X", where the orderable
            # part is VL53L1CXV0FY/1 — NOT a superstring of the family name).
            # Widen with progressively shortened prefixes until in-stock rows
            # join the candidate set; bounded to keep worst-case scans cheap.
            rows = list(rows)
            seen = {r["lcsc"] for r in rows}
            for probe in (q[:n] for n in range(len(q), len(q) - 4, -1)):
                if len(probe) < 4:
                    break
                extra = con.execute(f"{sel} WHERE mfr LIKE ? {order}",
                                    (f"%{probe}%", limit)).fetchall()
                in_stock = [r for r in extra
                            if r["stock"] > 0 and r["lcsc"] not in seen]
                if in_stock:
                    rows += in_stock
                    rows.sort(key=lambda r: -r["stock"])
                    break
        if not rows:
            rows = con.execute(f"{sel} WHERE mfr LIKE ? {order}",
                               (f"%{q}%", limit)).fetchall()
        if not rows:
            terms = [t for t in re.split(r"\s+", q) if len(t) >= 2]
            if terms:
                hay = "(mfr || ' ' || description || ' ' || manufacturer || ' ' || package)"
                cond = " AND ".join([f"{hay} LIKE ?"] * len(terms))
                rows = con.execute(f"{sel} WHERE {cond} {order}",
                                   [f"%{t}%" for t in terms] + [limit]).fetchall()
    finally:
        con.close()
    return [_candidate(r) for r in rows]


def lookup(lcsc_id: str | int) -> dict | None:
    """One part by LCSC id ('C190004' or 190004), with its full price ladder."""
    if not available():
        return None
    try:
        num = int(str(lcsc_id).strip().upper().lstrip("C"))
    except ValueError:
        return None
    con = _connect()
    try:
        row = con.execute(
            f"SELECT {_CANDIDATE_COLS} FROM jlc_components WHERE lcsc = ?",
            (num,)).fetchone()
    finally:
        con.close()
    if row is None:
        return None
    cand = _candidate(row)
    cand["ladder"] = parse_ladder(row["price"])
    return cand


# ----------------------------------------------- split-zip (zip -s) extraction
# The dump ships as cache.z01..cache.zNN + cache.zip (the last volume, holding
# the central directory). Python's zipfile refuses multi-disk archives and the
# host has no zip/unzip/7z, so this is a minimal stdlib extractor: resolve each
# entry's absolute offset from the per-disk offsets + cumulative volume sizes,
# then stream-inflate. Handles the ZIP64 fields the >4 GiB sqlite needs.

class _SpanReader:
    """Seek/read across the concatenation of the volume files."""

    def __init__(self, paths: list[Path]):
        self._paths = paths
        self.sizes = [p.stat().st_size for p in paths]
        self._starts = [sum(self.sizes[:i]) for i in range(len(paths))]
        self.total = sum(self.sizes)
        self._pos = 0

    def seek(self, pos: int) -> None:
        self._pos = pos

    def read(self, n: int) -> bytes:
        out = b""
        while n > 0 and self._pos < self.total:
            i = max(j for j, s in enumerate(self._starts) if s <= self._pos)
            local = self._pos - self._starts[i]
            with open(self._paths[i], "rb") as fh:
                fh.seek(local)
                chunk = fh.read(min(n, self.sizes[i] - local))
            if not chunk:
                break
            out += chunk
            self._pos += len(chunk)
            n -= len(chunk)
        return out


def _zip64_fields(extra: bytes, wanted: list[str], entry: dict) -> None:
    """Fill sentinel-valued `wanted` fields of `entry` from a ZIP64 extra block."""
    p = 0
    while p + 4 <= len(extra):
        tag, size = struct.unpack("<HH", extra[p:p + 4])
        body = extra[p + 4:p + 4 + size]
        if tag == 0x0001:
            q = 0
            for field in wanted:
                width = 4 if field == "disk_start" else 8
                if q + width > len(body):
                    break
                entry[field] = int.from_bytes(body[q:q + width], "little")
                q += width
        p += 4 + size


def _central_directory(reader: _SpanReader) -> list[dict]:
    tail_len = min(reader.total, 66_000)
    reader.seek(reader.total - tail_len)
    tail = reader.read(tail_len)
    e = tail.rfind(b"PK\x05\x06")
    if e < 0:
        raise ValueError("no end-of-central-directory record (not a zip?)")
    _, cd_disk, _, n_total, _, cd_off = struct.unpack("<HHHHII", tail[e + 4:e + 20])
    starts = [sum(reader.sizes[:i]) for i in range(len(reader.sizes))]
    if 0xFFFF in (cd_disk, n_total) or cd_off == 0xFFFFFFFF:
        loc = tail.rfind(b"PK\x06\x07", 0, e)
        if loc < 0:
            raise ValueError("ZIP64 sentinel without a ZIP64 locator")
        e64_disk, e64_off = struct.unpack("<IQ", tail[loc + 4:loc + 16])
        reader.seek(starts[e64_disk] + e64_off)
        e64 = reader.read(56)
        if e64[:4] != b"PK\x06\x06":
            raise ValueError("bad ZIP64 end-of-central-directory record")
        cd_disk = struct.unpack("<I", e64[20:24])[0]
        n_total = struct.unpack("<Q", e64[32:40])[0]
        cd_off = struct.unpack("<Q", e64[48:56])[0]

    reader.seek(starts[cd_disk] + cd_off)
    # The CD is tiny (one entry per member file); reading 1 MB is plenty.
    cd = reader.read(1 << 20)
    entries, p = [], 0
    for _ in range(n_total):
        if cd[p:p + 4] != b"PK\x01\x02":
            raise ValueError("corrupt central directory")
        (_, _, _, method, _, _, _, csize, usize, nlen, elen, clen,
         disk_start, _, _, local_off) = struct.unpack("<HHHHHHIIIHHHHHII",
                                                      cd[p + 4:p + 46])
        entry = {"name": cd[p + 46:p + 46 + nlen].decode("utf-8"),
                 "method": method, "csize": csize, "usize": usize,
                 "disk_start": disk_start, "local_off": local_off}
        sentinel = {"usize": 0xFFFFFFFF, "csize": 0xFFFFFFFF,
                    "local_off": 0xFFFFFFFF, "disk_start": 0xFFFF}
        wanted = [f for f, s in sentinel.items() if entry[f] == s]
        if wanted:
            _zip64_fields(cd[p + 46 + nlen:p + 46 + nlen + elen], wanted, entry)
        entry["abs_off"] = starts[entry["disk_start"]] + entry["local_off"]
        entries.append(entry)
        p += 46 + nlen + elen + clen
    return entries


def extract_split_zip(volumes: list[Path], dest_dir: Path) -> list[Path]:
    """Extract every member of a (possibly split) zip into dest_dir.
    `volumes` must be in order (.z01, .z02, ..., .zip last)."""
    reader = _SpanReader(volumes)
    out_paths = []
    for entry in _central_directory(reader):
        reader.seek(entry["abs_off"])
        lh = reader.read(30)
        if lh[:4] != b"PK\x03\x04":
            raise ValueError(f"bad local header for {entry['name']}")
        nlen, elen = struct.unpack("<HH", lh[26:30])
        reader.seek(entry["abs_off"] + 30 + nlen + elen)
        out = dest_dir / Path(entry["name"]).name   # flat: no path traversal
        inflater = zlib.decompressobj(-15) if entry["method"] == 8 else None
        if inflater is None and entry["method"] != 0:
            raise ValueError(f"unsupported compression method {entry['method']}")
        remaining = entry["csize"]
        with open(out, "wb") as fh:
            while remaining > 0:
                chunk = reader.read(min(1 << 22, remaining))
                if not chunk:
                    raise ValueError(f"truncated archive at {entry['name']}")
                remaining -= len(chunk)
                fh.write(inflater.decompress(chunk) if inflater else chunk)
            if inflater:
                fh.write(inflater.flush())
        out_paths.append(out)
    return out_paths


# ------------------------------------------------------------------- updater

def _download(url: str, dest: Path) -> bool:
    """Fetch url -> dest. False (not an error) when the resource is absent
    (HTTP 404, or a missing file for file:// test URLs), which is how the
    volume probe finds the end of the .zNN series."""
    req = urllib.request.Request(url, headers={"User-Agent": "kicraft"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
    except urllib.error.URLError as e:
        if isinstance(getattr(e, "reason", None), FileNotFoundError):
            return False
        raise


def update(dest: Path | None = None, base_url: str = DATA_URL,
           progress=lambda msg: None) -> dict:
    """Download the jlcparts dump, extract, index, and atomically install it.

    Returns {"db", "rows", "bytes"}. Raises on any failure, leaving the
    previous catalog (if any) untouched.
    """
    dest = dest or db_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / "update.tmp"
    tmp.mkdir(exist_ok=True)
    try:
        volumes = []
        for n in range(1, 100):
            name = f"cache.z{n:02d}"
            path = tmp / name
            if not _download(base_url + name, path):
                path.unlink(missing_ok=True)
                break
            volumes.append(path)
            progress(f"downloaded {name} ({path.stat().st_size / 1e6:.0f} MB)")
        last = tmp / "cache.zip"
        if not _download(base_url + "cache.zip", last):
            raise RuntimeError(f"{base_url}cache.zip not found")
        volumes.append(last)
        progress(f"downloaded cache.zip ({last.stat().st_size / 1e6:.0f} MB)")

        progress("extracting...")
        extracted = extract_split_zip(volumes, tmp)
        db_file = next((p for p in extracted if p.suffix == ".sqlite3"), None)
        if db_file is None:
            raise RuntimeError(f"no .sqlite3 member in the archive: {extracted}")

        con = sqlite3.connect(db_file)
        try:
            rows = con.execute("SELECT COUNT(*) FROM jlc_components").fetchone()[0]
            if rows < 100_000:
                raise RuntimeError(f"catalog looks wrong: only {rows} components")
            progress(f"indexing {rows:,} components...")
            con.execute("CREATE INDEX IF NOT EXISTS idx_jlc_mfr "
                        "ON jlc_components(mfr COLLATE NOCASE)")
            con.commit()
        finally:
            con.close()

        size = db_file.stat().st_size
        os.replace(db_file, dest)
        progress(f"installed {dest} ({size / 1e9:.2f} GB, {rows:,} parts)")
        return {"db": str(dest), "rows": rows, "bytes": size}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
