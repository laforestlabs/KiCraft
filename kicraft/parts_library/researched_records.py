"""The reviewed records this machine researched for itself.

A design can demand a part class the vendored records cannot answer -- a brief asks for a series
Schottky diode and the library carries none. That is not a dead end and must not be one: the
pipeline researches a real, orderable part for the class, vendors its symbol and footprint into
the HOME parts library, and writes the reviewed record here, so **every later run (and every other
project on this machine) already has it**. The cost is paid once.

The vendored records stay where they are, in `kicraft.design.part_identity.REVIEWED_PARTS`; this
file holds the machine's own additions and is their only writer. Stored as plain JSON so a record
can be read, diffed and hand-corrected without touching code -- a record that turns out wrong is
fixed by editing this file, not by editing a Python literal.

Layout, one object per record::

    {"identity": "b5819w", "family": "schottky-diode", "package": "SOD-123", ...,
     "operating_limits": {"reverse_voltage_v": 40.0, "rectified_current_a": 1.0},
     "limits_source": {"reverse_voltage_v": "catalog parameter 'Voltage - DC Reverse(Vr)' = '40V'"},
     "limits_review": {"status": "catalog-parameters", "source": "jlcparts (...) parametric fields",
                       "reason": "2 rating(s) read from the catalog's own parametric fields ..."},
     "unrated_parameters": {"Reverse Leakage Current (Ir)": "1mA@30V"},
     "researched": {"on": "2026-09-25", "from": "jlcparts", "lcsc": "C7420330",
                    "why": "the demanded class schottky-diode had no reviewed carrier"}}

Path: ``~/.kicraft/researched_records.json`` (``KICRAFT_RESEARCHED_RECORDS`` overrides it, which
is how the tests point at a scratch file), beside the other per-machine caches.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from kicraft.fsutil import atomic_write_text

#: Fields a stored record may carry, mirroring the reviewed-part shape. Unknown keys are kept as
#: written so a hand-edited record never loses facts a newer/older reader does not know about.
_LIST_FIELDS = ("physical_features", "function_keys", "contacts", "manufacturer_sources")


def records_path() -> Path:
    override = os.environ.get("KICRAFT_RESEARCHED_RECORDS")
    if override:
        return Path(override)
    return Path.home() / ".kicraft" / "researched_records.json"


def _as_dicts(raw: object) -> list[dict]:
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if isinstance(raw, dict):
        # Tolerate an object keyed by identity, which is easy to hand-write.
        return [{**row, "identity": str(row.get("identity") or key)}
                for key, row in raw.items() if isinstance(row, dict)]
    return []


def load(path: Path | None = None) -> tuple[dict, ...]:
    """Every stored record, oldest first. A missing or unreadable file is simply no records."""
    target = path or records_path()
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return ()
    try:
        return tuple(_as_dicts(json.loads(text)))
    except json.JSONDecodeError:
        # A half-written file must not take the pipeline down with it: the records it held are
        # re-researched on demand, which is slower but correct.
        return ()


def _canonical_row(record: Mapping[str, object]) -> dict:
    row = dict(record)
    for field in _LIST_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, (str, bytes)):
            row[field] = [value]
        else:
            row[field] = sorted({str(item) for item in value})  # type: ignore[union-attr]
    return row


def identity_key(record: Mapping[str, object]) -> tuple[str, str]:
    """How two records are recognised as the same part: its order code, or its LCSC id."""
    return (
        str(record.get("identity") or "").strip().casefold(),
        str(record.get("lcsc") or "").strip().upper(),
    )


def append(record: Mapping[str, object], path: Path | None = None) -> Path:
    """Store one researched record, replacing any earlier record of the same part.

    Replace rather than refuse: a second research pass over the same part (a corrected package, a
    better match) should leave one record per part, and the newest facts win.
    """
    target = path or records_path()
    row = _canonical_row(record)
    if not row.get("identity") or not row.get("family"):
        raise ValueError("a researched record needs at least an identity and a family")

    existing = list(load(target))
    key = identity_key(row)
    if not key[0] and not key[1]:
        raise ValueError("a researched record needs an identity or an LCSC id")
    kept = [other for other in existing if identity_key(other) != key]
    kept.append(row)

    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json.dumps(kept, indent=2, sort_keys=True) + "\n")
    return target


def coverage(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Every class the stored records claim, mapped to the order codes that claim it.

    The pipeline asks this when it wants to know whether a demand is already answered without
    re-reading the records: the keys are the physical features (the classes) and the values are
    the identities carrying them.
    """
    claims: dict[str, list[str]] = {}
    for record in load(path):
        features: Iterable[str] = record.get("physical_features") or ()
        if isinstance(features, str):
            features = (features,)
        for feature in features:
            claims.setdefault(str(feature).strip().casefold(), []).append(
                str(record.get("identity") or "")
            )
    return {feature: tuple(sorted(set(identities))) for feature, identities in claims.items()}
