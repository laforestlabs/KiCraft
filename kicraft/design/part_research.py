"""Research a real part for a class the reviewed library cannot answer.

A brief can demand a physical class no vendored record carries -- a series Schottky diode, a part
nobody has needed before. The honest answer is a real, orderable part: searched, ranked on
evidence, vendored, and **recorded once**, so every later run and every other project on this
machine already has it. That is this module: the loop the pipeline runs instead of refusing.

What each step claims:

- **the search** reads the offline JLC/LCSC catalog (`kicraft.parts_library.jlcparts`), so ranking
  is deterministic, free and works without a network round trip per candidate;
- **the class match** is the catalog's own description naming the demanded class's words, plus a
  single-device test (`jlcparts.is_multi_element_array`) -- evidence, quoted into the record;
- **the vendoring** runs the same CLI a stage tool runs (`add-part --from-lcsc <C> --into home`),
  so the bundle lands in the machine-wide HOME library with its manifest, symbol and footprint;
- **the record** carries the order code, the LCSC id, the exact symbol/footprint pair, the contacts
  read out of that symbol, the catalog description and the datasheet link. Its `operating_limits`
  stay **empty on purpose**: a rating this loop has not read off a datasheet must not be invented,
  and `ReviewedPart`'s convention is that an absent limit is unproven rather than generous.

Every entry point takes its searcher/fetcher/pins-reader as an argument, so the whole loop is
testable without a network, a catalog dump or KiCad libraries.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from kicraft.parts_library import jlcparts, researched_records

#: A researched part must clear the same sourcing floor the BOM gate enforces (`<100` is refused),
#: in both inventories. Overridable for a shop with different rules.
DEFAULT_MIN_STOCK = 100

#: How `add-part` is reached: the same invocation the stage tools use
#: (`kicraft.server.stage_state_io.KICRAFT`), spelled here so this design-layer module needs no
#: server import.
_CLI = (sys.executable, "-m", "kicraft.design.cli_app")

#: Tokens too generic to prove a class match on their own.
_WEAK_TOKENS = frozenset({"smd", "smt", "tht", "mini", "micro", "chip", "single", "dual", "of",
                          "and", "the", "a", "for", "with", "high", "low", "power"})


@dataclass(frozen=True)
class ResearchResult:
    """One successfully researched class: what was chosen, and whether this call vendored it."""

    component_class: str
    identity: str
    lcsc: str | None
    symbol: str | None
    footprint: str | None
    fetched: bool
    reason: str

    def as_record_note(self) -> str:
        """The one-line record a stage can carry as an assumption, in plain words."""
        return (
            f"the demanded class {self.component_class!r} had no placeable part, so {self.identity}"
            f"{f' (LCSC {self.lcsc})' if self.lcsc else ''} was researched and added to the parts "
            "library, where it stays for later runs (defaulted)"
        )


# ------------------------------------------------------------------ searching


def class_keywords(component_class: str, extra: Iterable[str] = ()) -> tuple[str, ...]:
    """Search phrases for a class: its own words, then any the caller adds.

    ``schottky-diode`` searches "schottky diode"; a caller that knows the design's constraints
    ("SOD-123", "40 V") appends them and the ranking prefers rows matching more of them.
    """
    words = [token for token in re.split(r"[^a-z0-9]+", component_class.casefold()) if token]
    phrases = [" ".join(words)] if words else []
    phrases.extend(str(item).strip() for item in extra if str(item).strip())
    return tuple(dict.fromkeys(phrases))


def _class_tokens(component_class: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", component_class.casefold())
        if len(token) >= 3 and token not in _WEAK_TOKENS
    }


def search_candidates(
    component_class: str,
    *,
    keywords: Iterable[str] = (),
    limit: int = 40,
    searcher: Callable[[str, int], list[dict]] | None = None,
) -> list[dict]:
    """Catalog rows matching the class's words, best-first per the search's own ordering.

    Deduplicated by LCSC id across phrases, so a row found by two phrases is one candidate.
    """
    search = searcher or (lambda query, cap: jlcparts.search(query, limit=cap))
    found: dict[str, dict] = {}
    for phrase in class_keywords(component_class, keywords):
        for row in search(phrase, limit) or ():
            if not isinstance(row, Mapping):
                continue
            lcsc = str(row.get("lcsc") or "").strip().upper()
            if not lcsc or lcsc in found:
                continue
            # Which phrase found it is evidence in its own right: the catalog matched that phrase
            # against the row's own text (mfr/description/manufacturer/package), and the text it
            # hands back is truncated, so a class word past the cut is invisible to a reader here.
            found[lcsc] = {**dict(row), "_matched_phrase": phrase}
    return list(found.values())


def _row_text(row: Mapping[str, object]) -> str:
    return " ".join(
        str(row.get(key) or "") for key in ("description", "package", "model", "brand")
    ).casefold()


def _row_stock(row: Mapping[str, object]) -> int:
    try:
        return int(row.get("stock") or 0)
    except (TypeError, ValueError):
        return 0


def _row_price(row: Mapping[str, object]) -> float:
    try:
        return float(row.get("price") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def choose_candidate(
    component_class: str,
    rows: Sequence[Mapping[str, object]],
    *,
    package: str | None = None,
    min_stock: int = DEFAULT_MIN_STOCK,
) -> tuple[dict | None, str]:
    """The best row for the class, with the reason it won -- or (None, why nothing qualified).

    Ranking is evidence first, then availability, then price: a row whose description names every
    class token beats one that names fewer; then the row that is *only* one device inside the
    package (a dual-diode array is not a diode); then the most stock, then the cheapest. Ties break
    on the LCSC id so the same catalog always yields the same choice.
    """
    tokens = _class_tokens(component_class)
    if not tokens:
        return None, f"the class {component_class!r} names no searchable token"

    scored: list[tuple[int, int, float, str, dict]] = []
    rejected = {"out_of_stock": 0, "wrong_package": 0, "multi_element": 0, "no_class_word": 0}
    for row in rows:
        candidate = dict(row)
        text = _row_text(candidate)
        phrase = str(candidate.get("_matched_phrase") or "").casefold()
        # The class's own phrase finding the row is evidence: the catalog matched those words
        # against the row's text, and only a truncated view of that text reaches this function.
        hits = sum(1 for token in tokens if token in text or token in phrase)
        if not hits:
            rejected["no_class_word"] += 1
            continue
        if _row_stock(candidate) < min_stock:
            rejected["out_of_stock"] += 1
            continue
        if package and package.casefold() not in text:
            rejected["wrong_package"] += 1
            continue
        try:
            if jlcparts.is_multi_element_array(candidate):
                rejected["multi_element"] += 1
                continue
        except Exception:  # noqa: BLE001 - the verdict is optional; never fail the search on it
            pass
        lcsc = str(candidate.get("lcsc") or "")
        scored.append((hits, _row_stock(candidate), -_row_price(candidate), lcsc, candidate))

    if not scored:
        why = ", ".join(
            f"{count} {reason.replace('_', ' ')}" for reason, count in rejected.items() if count
        )
        return None, f"no catalog row for {component_class!r} qualified ({why or 'no rows'})"

    hits, stock, _neg_price, _lcsc, best = max(scored, key=lambda item: item[:4])
    matched = sorted(token for token in tokens if token in _row_text(best))
    return best, (
        f"{best.get('model') or best.get('lcsc')} matches {', '.join(matched)} in its own "
        f"description, is in stock ({stock}) and is a single-device {component_class}"
    )


# ------------------------------------------------------------------ vendoring


def _slug(component_class: str, lcsc: str) -> str:
    words = re.sub(r"[^a-z0-9]+", "-", component_class.casefold()).strip("-")
    return f"{(words or 'part')[:24]}-{lcsc.casefold()}"


def fetch_bundle(lcsc_id: str, slug: str, *, component_class: str = "", description: str = "") -> dict | None:
    """Vendor one part's symbol+footprint into the HOME library; return its manifest, or None.

    Runs the same command a stage tool runs, then reads the manifest the CLI just wrote -- no
    parsing of console output, and the bundle lands where every project on this machine looks.

    The bundle's own maturity stays the CLI's default (`prototype`: auto-fetched, not human-vetted).
    That is the honest badge for a machine-researched part, and the record says the same thing in
    its `researched` block; tagging it with the class is what makes it findable by class later.
    """
    from kicraft.parts_library.loader import home_parts_dir

    command = [*_CLI, "add-part", "--from-lcsc", lcsc_id, "--into", "home", "--name", slug]
    if component_class:
        command += ["--tag", component_class]
    if description:
        command += ["--description", description[:120]]
    completed = subprocess.run(
        command, capture_output=True, text=True, cwd=os.getcwd()
    )
    if completed.returncode != 0:
        return None
    manifest = home_parts_dir() / slug / "manifest.json"
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def symbol_contacts(symbol: str) -> tuple[str, ...]:
    """The contact numbers the vendored symbol publishes, or none when it cannot be read."""
    try:
        from kicraft.design.synthesis.symbol_pinout import lookup_pins

        info = lookup_pins(symbol, all_units=True)
    except Exception:  # noqa: BLE001 - a record without contacts is still useful
        return ()
    pins = info.get("pins") if isinstance(info, Mapping) else None
    rows = pins.values() if isinstance(pins, Mapping) else pins or ()
    numbers: list[str] = []
    for pin in rows:
        number = pin.get("number") if isinstance(pin, Mapping) else getattr(pin, "number", None)
        if number is not None:
            numbers.append(str(number))
    return tuple(dict.fromkeys(numbers))


def build_record(
    component_class: str,
    candidate: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    contacts: Sequence[str] = (),
    today: str | None = None,
) -> dict:
    """The stored record for one researched part: what it is, why it was chosen, where it came from."""
    slug = str(manifest.get("name") or "")
    symbol_name = str(manifest.get("symbol_name") or "")
    footprint_name = str(manifest.get("footprint_name") or "")
    identity = str(manifest.get("mpn") or candidate.get("model") or "").strip()
    lcsc = str((manifest.get("sourcing") or {}).get("lcsc") or candidate.get("lcsc") or "")
    sources = [
        url
        for url in (
            manifest.get("datasheet_url"),
            (f"https://lcsc.com/product-detail/{lcsc}.html" if lcsc else None),
        )
        if url
    ]
    return {
        "identity": identity.casefold(),
        "family": component_class.strip().casefold(),
        "package": str(
            manifest.get("description") or candidate.get("package") or candidate.get("description") or ""
        ),
        "bundle": slug or None,
        "symbol": f"{slug}:{symbol_name}" if slug and symbol_name else None,
        "footprint": f"{slug}:{footprint_name}" if slug and footprint_name else None,
        "physical_features": [component_class.strip().casefold()],
        "contacts": list(contacts),
        "manufacturer_sources": sources,
        "lcsc": lcsc or None,
        "researched": {
            "on": today or date.today().isoformat(),
            "from": "jlcparts (offline JLC/LCSC catalog)",
            "lcsc": lcsc or None,
            "catalog_description": str(candidate.get("description") or ""),
            "why": (
                f"the demanded class {component_class!r} had no placeable reviewed part; this part's "
                "catalog description names the class and it is in stock"
            ),
            "limits": "not reviewed: no rating was read off a datasheet, so none is claimed",
        },
    }


# ------------------------------------------------------------------ the loop


#: One research per class per process: a stage that redrafts must not refetch the same part.
_RESEARCHED_THIS_PROCESS: dict[str, ResearchResult | None] = {}


def research_part_for_class(
    component_class: str,
    *,
    keywords: Iterable[str] = (),
    package: str | None = None,
    min_stock: int = DEFAULT_MIN_STOCK,
    searcher: Callable[[str, int], list[dict]] | None = None,
    fetcher: Callable[..., Mapping[str, object] | None] | None = None,
    pins_reader: Callable[[str], Sequence[str]] | None = None,
    records_path: Path | None = None,
) -> ResearchResult | None:
    """Research and record a part for one demanded class. ``None`` when nothing needed doing.

    Returns ``None`` when the class already has a placeable reviewed part (the common case, and the
    reason this is cheap to call from a pipeline), or when the search found nothing this loop is
    willing to stand behind -- in which case the class stays unanswered and the caller's own
    refusal speaks, rather than a wrong part being smuggled in.
    """
    from kicraft.design.part_identity import reviewed_parts_for_feature

    key = component_class.strip().casefold()
    if not key:
        return None
    if reviewed_parts_for_feature(key):
        return None
    if key in _RESEARCHED_THIS_PROCESS:
        return _RESEARCHED_THIS_PROCESS[key]

    rows = search_candidates(key, keywords=keywords, searcher=searcher)
    chosen, reason = choose_candidate(key, rows, package=package, min_stock=min_stock)
    if chosen is None:
        _RESEARCHED_THIS_PROCESS[key] = None
        return None

    lcsc = str(chosen.get("lcsc") or "").strip().upper()
    slug = _slug(key, lcsc)
    fetch = fetcher or fetch_bundle
    # The fetcher protocol: (lcsc_id, slug, component_class=..., description=...) -> manifest|None.
    manifest = fetch(
        lcsc, slug, component_class=key, description=str(chosen.get("description") or "")
    )
    if not manifest:
        _RESEARCHED_THIS_PROCESS[key] = None
        return None

    record = build_record(
        key,
        chosen,
        manifest,
        contacts=(pins_reader or symbol_contacts)(
            f"{slug}:{manifest.get('symbol_name')}" if manifest.get("symbol_name") else ""
        ),
    )
    researched_records.append(record, records_path)

    result = ResearchResult(
        component_class=key,
        identity=str(record["identity"]),
        lcsc=record.get("lcsc"),
        symbol=record.get("symbol"),
        footprint=record.get("footprint"),
        fetched=True,
        reason=reason,
    )
    _RESEARCHED_THIS_PROCESS[key] = result
    return result


def research_uncovered_classes(
    component_classes: Iterable[str],
    *,
    limit: int = 2,
    **kwargs,
) -> list[ResearchResult]:
    """Research the first ``limit`` uncovered classes from a demand list.

    A stage can demand several unknown classes at once; the cap keeps one draft from turning into
    an unbounded fetching run. Classes beyond it stay unanswered for a later pass.
    """
    done: list[ResearchResult] = []
    seen: set[str] = set()
    for component_class in component_classes:
        key = str(component_class or "").strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        if len(done) >= limit:
            break
        result = research_part_for_class(key, **kwargs)
        if result is not None:
            done.append(result)
    return done
