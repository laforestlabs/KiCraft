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
  read out of that symbol, the catalog description and the datasheet link;
- **the ratings** are the catalog's own parametric fields, quoted: `"Voltage - DC Reverse(Vr)"` =
  `"40V"` becomes the record's `reverse_voltage_v: 40.0`, and every limit kept names in
  `limits_source` the catalog parameter it came from. Nothing is read off a value that does not
  state it -- a range becomes a pair only where the pipeline already names one, a lone number that
  could be a nominal becomes no rating at all -- and every parameter no rating is claimed from is
  kept verbatim in `unrated_parameters`. A record that reads no rating says so: `"unverified"`,
  never a blank field that could pass for "no limit needed".

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
from dataclasses import dataclass, field
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


# ------------------------------------------------------------------ ratings
#
# A researched part has no hand reading from a datasheet, so the honest rating is the catalog's
# own parametric field, quoted. The names below are the ones the JLC/LCSC dump actually publishes
# (`kicraft.parts_library.jlcparts.parameters`), mapped onto the limit names this pipeline's
# reviewed records already use, so a researched part is read by the same checks as a hand-reviewed
# one. Only parameters a check can act on are mapped; the rest are kept verbatim instead.

#: One catalog parameter -> the rating it publishes. `(limit key, quantity kind)`.
_CATALOG_RATINGS: Mapping[str, tuple[str, str]] = {
    # Diodes: Schottky, switching, zener, TVS/ESD, varistor
    "voltage dc reverse vr": ("reverse_voltage_v", "volts"),
    "current rectified": ("rectified_current_a", "amps"),
    "voltage forward vf": ("forward_voltage_v", "volts"),
    "voltage forward vf if": ("forward_voltage_v", "volts"),
    "forward current": ("forward_current_ma", "milliamps"),
    "zener voltage nom": ("zener_voltage_v", "volts"),
    "reverse stand off voltage vrwm": ("reverse_standoff_voltage_v", "volts"),
    "clamping voltage": ("clamping_voltage_v", "volts"),
    "maximum dc volts": ("max_dc_volts_v", "volts"),
    "maximum ac volts": ("max_ac_volts_v", "volts"),
    # Transistors
    "drain to source voltage": ("vds_max_v", "volts"),
    "current continuous drain id": ("continuous_drain_a", "amps"),
    "rds on": ("rds_on_ohm", "ohms"),
    "collector emitter voltage vceo": ("vceo_max_v", "volts"),
    "current collector ic": ("collector_current_a", "amps"),
    # Passives
    "capacitance": ("capacitance_f", "farads"),
    "inductance": ("inductance_h", "henries"),
    "dc resistance dcr": ("dc_resistance_ohm", "ohms"),
    "resistance": ("resistance_ohm", "ohms"),
    "current rating": ("current_a", "amps"),
    "current saturation isat": ("saturation_current_a", "amps"),
    "power watts": ("power_w", "watts"),
    "pd power dissipation": ("power_dissipation_w", "watts"),
    # Connectors, headers, sockets, terminals
    "voltage rating": ("voltage_v", "volts"),
    "voltage rating max": ("voltage_v", "volts"),
    # A switch's contact current is the same single rating a reviewed switch record carries as
    # `current_a`, and a fuse's DC voltage rating the same one a jack/relay record carries as
    # `voltage_vdc`; the AC rating is a different fact and is left unclaimed.
    "contact current": ("current_a", "amps"),
    "voltage rating dc": ("voltage_vdc", "volts"),
    # A chip resistor's "Voltage-Supply(Max)" is the working voltage across the part, not a supply
    # domain: it is the same single rating a reviewed chip-passive record carries as `voltage_v`.
    "voltage supply max": ("voltage_v", "volts"),
    "pitch": ("pitch_mm", "millimetres"),
    "number of pins": ("positions", "count"),
    "number of positions": ("positions", "count"),
    # Regulators, converters and driven parts
    "output voltage": ("output_voltage_v", "volts"),
    "output current": ("output_current_a", "amps"),
    "current consumption": ("current_consumption_ma", "milliamps"),
    "frequency": ("frequency_hz", "hertz"),
    "load capacitance": ("load_capacitance_f", "farads"),
}

#: Catalog parameters that publish a *range*, and the pair of limit names that carries one.
#: A range is claimed as a pair because the catalog states both ends; a part whose parameter here
#: holds a lone number gets no rating: "Voltage - Supply: 3.3V" on an oscillator is a nominal and
#: "30V" on a regulator is a maximum, and nothing in the field says which, so a guess would be a
#: fabricated limit. Claiming the pair is also what keeps the input-range check honest: it refuses
#: a record that states a voltage input without both ends, and a researched record declares no
#: input pin, so a lone maximum would turn a good part into a refusal.
_CATALOG_RANGES: Mapping[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "voltage supply": (("supply_min_v", "volts"), ("supply_max_v", "volts")),
    "operating temperature": (("temperature_min_c", "celsius"), ("temperature_max_c", "celsius")),
    "operating junction temperature range": (
        ("temperature_min_c", "celsius"),
        ("temperature_max_c", "celsius"),
    ),
}

#: Where a rating on a researched record came from.
_CATALOG_SOURCE = "jlcparts (offline JLC/LCSC catalog) parametric fields"

#: Multipliers, case-sensitive on purpose: "MHz" is megahertz and "mHz" is not.
_SI_PREFIX: Mapping[str, float] = {
    "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3,
    "": 1.0, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9,
}

#: Unit spellings the catalog uses -> the quantity kind the value is. Longest base first: "mΩ"
#: ends with "Ω", "kHz" with "Hz", and "ppm/℃" matches nothing.
_BASE_UNITS: tuple[tuple[str, str], ...] = (
    ("ohm", "ohms"), ("Ω", "ohms"), ("v", "volts"), ("a", "amps"), ("w", "watts"),
    ("f", "farads"), ("h", "henries"), ("hz", "hertz"), ("c", "celsius"), ("j", "joules"),
)

_QUANTITY_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([A-Za-zµμΩ℃°%]*)$")


def _parameter_key(name: object) -> str:
    """A catalog parameter's name, reduced to words: `"Voltage - DC Reverse(Vr)"` becomes
    `"voltage dc reverse vr"` and `"Pd - Power Dissipation"` becomes `"pd power dissipation"`."""
    return " ".join(re.split(r"[^a-z0-9]+", str(name).casefold())).strip()


def _quantity(text: object) -> tuple[float, str] | None:
    """`"12mΩ@10V"` -> `(0.012, "ohms")`; a condition after `@` is the test point, not the rating.

    `None` when the text is not exactly one number with one unit this loop recognises -- a phrase,
    a bare tolerance, two magnitudes in one field. The caller records those verbatim instead.
    """
    if text is None or isinstance(text, bool):
        return None
    head = str(text).split("@", 1)[0].strip()
    if head[:1] == "±":  # the regex already carries a signed number; only "±" is not one
        head = head[1:].strip()
    match = _QUANTITY_RE.match(head)
    if match is None:
        return None
    number = float(match.group(1))
    token = match.group(2)
    if not token:
        return number, "count"
    if token == "%":
        return number, "percent"
    if token.casefold() == "mm":
        return number, "millimetres"
    if token.casefold() in ("p", "pin", "pins", "pos", "positions"):
        return number, "count"
    token = token.replace("℃", "°C").lstrip("°")  # "℃"/"120°C" are the same unit spelled two ways
    if not token:
        return None
    for base, kind in _BASE_UNITS:
        # `<` and not `<=`: a bare unit ("40V", "-40℃") is one character and must still match.
        if len(token) < len(base) or token[len(token) - len(base):].casefold() != base.casefold():
            continue
        scale = _SI_PREFIX.get(token[: len(token) - len(base)])
        return None if scale is None else (number * scale, kind)
    return None


def _range_quantities(text: object) -> tuple[tuple[float, str], tuple[float, str]] | None:
    """`"-55℃~+155℃"` -> `((-55.0, "celsius"), (155.0, "celsius"))`; `None` when one value.

    One end of the catalog's range sometimes drops the unit (`"-55~85℃"`); it takes the other
    end's, because the two ends of one range are one quantity. Two bare numbers (`"2~6"`) are left
    unread: nothing in them says what they count in.
    """
    head = str(text or "").split("@", 1)[0].strip()
    parts = [part.strip() for part in re.split(r"[~～]", head)]
    if len(parts) != 2:
        return None
    low, high = _quantity(parts[0]), _quantity(parts[1])
    if low is None or high is None:
        return None
    if low[1] == "count" and high[1] != "count":
        low = (low[0], high[1])
    elif high[1] == "count" and low[1] != "count":
        high = (high[0], low[1])
    if low[1] != high[1]:
        return None
    return (low, high) if low[0] <= high[0] else (high, low)


#: Rating names the pipeline's own records state in a smaller unit than the SI base, and the
#: factor that takes the base unit to it ("forward_current_ma": 20, not 0.02).
_RATING_UNITS: Mapping[str, tuple[str, float]] = {
    "milliamps": ("amps", 1000.0),
}


def _as_rating(measured: tuple[float, str] | None, kind: str) -> float | None:
    """The measured quantity in the unit the rating name uses, or `None` when it is another kind."""
    if measured is None:
        return None
    if measured[1] == kind:
        return measured[0]
    base, factor = _RATING_UNITS.get(kind, (None, None))
    return measured[0] * factor if base is not None and measured[1] == base else None


def catalog_ratings(
    parameters: Mapping[str, object] | None,
) -> tuple[dict[str, float], dict[str, str], dict[str, object]]:
    """`(ratings, sources, unrated)` for one part's catalog parametric fields.

    ``ratings`` are the limit names the pipeline reads, ``sources`` names the catalog parameter and
    raw value behind each one, and ``unrated`` is every parameter left over verbatim -- so what
    could not be turned into a rating is kept, not dropped.
    """
    ratings: dict[str, float] = {}
    sources: dict[str, str] = {}
    unrated: dict[str, object] = {}
    for name, raw in (parameters or {}).items():
        key = _parameter_key(name)
        single = _CATALOG_RATINGS.get(key)
        if single is not None:
            limit_key, kind = single
            value = _as_rating(_quantity(raw), kind)
            if value is not None:
                ratings[limit_key] = round(value, 12)
                sources[limit_key] = f"catalog parameter {str(name)!r} = {str(raw)!r}"
                continue
        spanned = _CATALOG_RANGES.get(key)
        if spanned is not None:
            measured = _range_quantities(raw)
            if measured is None:
                unrated[str(name)] = raw
                continue
            (min_key, min_kind), (max_key, max_kind) = spanned
            low = _as_rating(measured[0], min_kind)
            high = _as_rating(measured[1], max_kind)
            if low is None or high is None:
                unrated[str(name)] = raw
                continue
            ratings[min_key] = round(low, 12)
            ratings[max_key] = round(high, 12)
            for limit_key in (min_key, max_key):
                sources[limit_key] = f"catalog parameter {str(name)!r} = {str(raw)!r}"
            continue
        unrated[str(name)] = raw
    return ratings, sources, unrated


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
    #: The ratings the catalog's own parametric fields state for this part, empty when it states
    #: none -- what the record now carries, so a caller can show them without re-reading the file.
    limits: Mapping[str, float] = field(default_factory=dict)

    def ratings_note(self) -> str:
        """The part's ratings in one plain line, or what the catalog left unstated."""
        if not self.limits:
            return "no rating the catalog states: every limit is unverified, not absent"
        return ", ".join(
            f"{key} {value:g}" for key, value in sorted(self.limits.items())
        )

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


def _catalog_parameters(lcsc: str) -> Mapping[str, object] | None:
    """The catalog's own parametric table for a part, or none when the dump lacks one.

    Imported here rather than at module scope for the same reason the CLI is: this design-layer
    module reaches the parts library, not the other way round.
    """
    from kicraft.parts_library import jlcparts

    try:
        return jlcparts.parameters(lcsc)
    except Exception:  # noqa: BLE001 - a record without ratings is still a record
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
    attributes: Mapping[str, object] | None = None,
    datasheet_url: str | None = None,
    today: str | None = None,
) -> dict:
    """The stored record for one researched part: what it is, why it was chosen, where it came from.

    ``attributes`` is the catalog's own parametric table for the part (see
    ``kicraft.parts_library.jlcparts.parameters``) and ``datasheet_url`` its catalog datasheet
    link. A rating is claimed only where one of those parameters states it; the parameters no
    rating is claimed from are kept verbatim under ``unrated_parameters``, and
    ``limits_review.status`` says plainly whether any rating was read at all.
    """
    slug = str(manifest.get("name") or "")
    symbol_name = str(manifest.get("symbol_name") or "")
    footprint_name = str(manifest.get("footprint_name") or "")
    identity = str(manifest.get("mpn") or candidate.get("model") or "").strip()
    lcsc = str((manifest.get("sourcing") or {}).get("lcsc") or candidate.get("lcsc") or "")
    sources = list(dict.fromkeys(
        url
        for url in (
            manifest.get("datasheet_url"),
            datasheet_url,
            (f"https://lcsc.com/product-detail/{lcsc}.html" if lcsc else None),
        )
        if url
    ))
    ratings, rating_sources, unrated = catalog_ratings(attributes)
    if ratings:
        limits_review = {
            "status": "catalog-parameters",
            "source": _CATALOG_SOURCE,
            "reason": (
                f"{len(ratings)} rating(s) read from the catalog's own parametric fields and named "
                "in limits_source; they are catalog data, not a datasheet or a human review, so "
                "every rating not listed there is unverified rather than absent"
            ),
        }
    else:
        limits_review = {
            "status": "unverified",
            "source": _CATALOG_SOURCE,
            "reason": (
                "the catalog publishes no parametric value this pipeline can read as a rating (or "
                "none at all for this part), so every rating is unverified rather than absent"
            ),
        }
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
        "operating_limits": ratings,
        "limits_source": rating_sources,
        "limits_review": limits_review,
        "unrated_parameters": unrated,
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
    parameters_reader: Callable[[str], Mapping[str, object] | None] | None = None,
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

    catalog = (parameters_reader or _catalog_parameters)(lcsc) or {}
    record = build_record(
        key,
        chosen,
        manifest,
        contacts=(pins_reader or symbol_contacts)(
            f"{slug}:{manifest.get('symbol_name')}" if manifest.get("symbol_name") else ""
        ),
        attributes=catalog.get("attributes"),
        datasheet_url=(
            str(catalog.get("datasheet")) if catalog.get("datasheet") else None
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
        limits=dict(record["operating_limits"]),
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
