"""What to do about a part that is real but out of stock.

The owner's rule (2026-09-26): a board must still get made when one or two of its parts cannot be
bought today. Three outcomes, decided here so every caller decides them the same way:

- **keep** a part the brief names by its own order code. The owner may hold stock or have a second
  source, and only they can weigh that. A person is asked when one is watching; an auto-defaulted
  run keeps the part and records the shortfall so the board is not held up by it.
- **swap** a part the pipeline chose for a class the brief named: an in-stock carrier of the *same
  family* **and the same package** takes its place, so every pin, footprint and wire the design
  already states stays valid. A different package is a different board (pins move), which is the
  owner's decision, never a silent substitution.
- **keep, with the reason**, when no such carrier exists -- and name what a different package would
  offer, because that is the choice left.

Nothing here refuses a board. Existence and identity are still hard gates elsewhere; this module is
only about a part that is real, matches the design, and cannot be bought this week.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

#: The two actions. `ask` is not a third action: the decision to keep is the same, and whether a
#: person is asked about it depends on the run (see ``StockDecision.asked``).
KEEP = "keep"
SWAP = "swap"


@dataclass(frozen=True)
class StockDecision:
    """What to do about one out-of-stock part, and the sentence that explains it."""

    action: str
    named_in_brief: bool
    #: True when a person should be the one to decide: the brief named the part, so it is theirs.
    asked: bool
    #: The reviewed carrier to use instead, when the action is `swap`.
    alternative: object | None
    reason: str


def _identity_key(text: object) -> str:
    """Order codes compare on letters and digits alone: `SOP-8_L4.9` vs `SOP-8 L4.9` is one code."""
    return re.sub(r"[^a-z0-9]+", "", str(text or "").casefold())


def part_number_named(
    identity: str,
    mpn: str | None,
    named_parts: Iterable[str] = (),
    brief: str = "",
) -> bool:
    """Whether the brief calls out this part's own *order code* (not just its family).

    A brief that says "a CH32V003 development board" names a family: the pipeline is free to pick
    the variant, and its choice may be swapped. A brief that says "CH32V003J4M6" names the part,
    and that one belongs to the owner -- they may have it on the shelf.
    """
    keys = {_identity_key(identity), _identity_key(mpn)} - {""}
    if not keys:
        return False
    named = {_identity_key(name) for name in named_parts} - {""}
    if keys & named:
        return True
    text = _identity_key(brief)
    return any(key in text for key in keys) if text else False


def _footprint_leaf(footprint: object) -> str:
    """A footprint without its library prefix: the package, which is what pins depend on."""
    return str(footprint or "").partition(":")[2] or str(footprint or "")


def same_package(record, other) -> bool:
    """Whether two reviewed carriers are the same package with the same contacts.

    The swap is only safe when the footprint *shape* and the contact count match: then every pad
    number the design already binds means the same thing in the replacement part. A different
    package (a SOP-16 standing in for a SOP-8) moves the pins, which is a design change.
    """
    return (
        _footprint_leaf(getattr(record, "footprint", "")) == _footprint_leaf(getattr(other, "footprint", ""))
        and tuple(getattr(record, "contacts", ()) or ()) == tuple(getattr(other, "contacts", ()) or ())
        and bool(getattr(record, "contacts", ()) or ())
    )


def swap_candidates(
    family: str,
    identity: str,
    *,
    in_stock: Callable[[object], bool],
    inventory: Iterable[object] | None = None,
) -> tuple[object, ...]:
    """In-stock reviewed carriers of ``family`` that share ``identity``'s package.

    Same family keeps the *device* class (a CH32V003 stays a CH32V003), and the same package keeps
    the pins; together they make the swap invisible to the rest of the design. Stock is the
    caller's reading (the two inventories live behind one callable), so this stays testable.
    """
    from kicraft.design.part_identity import reviewed_inventory, reviewed_part

    key = str(family or "").strip().casefold()
    current = reviewed_part(identity)
    if not key or current is None:
        return ()
    rows = inventory if inventory is not None else reviewed_inventory()
    out = []
    for candidate in rows:
        if not getattr(candidate, "is_portable_candidate", False):
            continue
        if str(getattr(candidate, "family", "")).strip().casefold() != key:
            continue
        if getattr(candidate, "identity", "") == getattr(current, "identity", ""):
            continue
        if not same_package(current, candidate):
            continue
        if in_stock(candidate):
            out.append(candidate)
    return tuple(sorted(out, key=lambda row: str(getattr(row, "identity", ""))))


def family_variants(family: str, identity: str) -> tuple[object, ...]:
    """Every other reviewed carrier of the family, whatever its package.

    Not a swap source: a different package moves the pins, so these are what the caller *names*
    when it reports that only a design change can help.
    """
    from kicraft.design.part_identity import reviewed_inventory, reviewed_part

    key = str(family or "").strip().casefold()
    current = reviewed_part(identity)
    if not key:
        return ()
    return tuple(
        sorted(
            (
                row
                for row in reviewed_inventory()
                if str(getattr(row, "family", "")).strip().casefold() == key
                and getattr(row, "identity", "") != getattr(current, "identity", "")
            ),
            key=lambda row: str(getattr(row, "identity", "")),
        )
    )


def stock_questions(
    parts: Iterable[object],
    *,
    named_parts: Iterable[str] = (),
    brief: str = "",
    stock: Callable[[str, object], tuple[bool, str]] | None = None,
    limit: int = 2,
) -> list[dict]:
    """The pipeline's own question for a part the brief names that cannot be bought today.

    Only a part the brief calls out by its own order code is asked about: it is the owner's part,
    so they decide between keeping it (their own stock, a second source, a wait) and taking an
    in-stock variation this board can accept unchanged. Unnamed parts are the pipeline's own choice
    and never need an answer -- they are swapped or kept by :func:`decide`.

    Returns question dicts in the shape the stage runtime parks (``text``/``blocking``/``options``),
    the first option being the default the automatic path takes: keep the part as designed.
    """
    reader = stock or _retail_and_catalog_stock
    out: list[dict] = []
    for part in parts:
        row = part if isinstance(part, Mapping) else vars(part)
        identity = str(row.get("mpn") or row.get("value") or "").strip()
        if not identity or not part_number_named(
            identity, row.get("mpn"), named_parts, brief
        ):
            continue
        record = _reviewed_record_for(row)
        cid = str((getattr(record, "lcsc", "") or _pinned_lcsc(row)) or "").strip().upper()
        if not cid:
            continue
        ok, shortfall = reader(cid, record)
        if ok:
            continue
        # The variations this board can take without a design change: same family, same package.
        options = [f"Keep {identity} as designed"]
        if record is not None:
            for candidate in swap_candidates(
                str(getattr(record, "family", "") or ""),
                str(getattr(record, "identity", "") or identity),
                in_stock=lambda row2: reader(str(getattr(row2, "lcsc", "") or ""), row2)[0],
            )[:limit]:
                options.append(
                    f"Use {getattr(candidate, 'identity', '')} "
                    f"(LCSC {getattr(candidate, 'lcsc', '')}, in stock)"
                )
            for variant in family_variants(
                str(getattr(record, "family", "") or ""),
                str(getattr(record, "identity", "") or identity),
            )[:limit]:
                cid2 = str(getattr(variant, "lcsc", "") or "")
                if cid2 and reader(cid2, variant)[0]:
                    options.append(
                        f"Change the design to {getattr(variant, 'identity', '')} "
                        f"(LCSC {cid2}) — a different package, so the design is opened again"
                    )
        options.append("Stop and let me supply or choose the part myself")
        out.append(
            {
                "text": (
                    f"{identity} (LCSC {cid}) is {shortfall}. The brief names this part, so it is "
                    "kept unless you say otherwise — do you want to build with it anyway, or "
                    "change it to something in stock?"
                ),
                "blocking": True,
                "material": True,
                "options": options[:4],
            }
        )
        if len(out) >= limit:
            break
    return out


def _pinned_lcsc(row: Mapping) -> str:
    """The C# a part pins in its sourcing note, if any."""
    from kicraft.design.synthesis.fab_export import extract_lcsc_pin

    return str(extract_lcsc_pin(str(row.get("sourcing_note") or "")) or "")


def _reviewed_record_for(row: Mapping):
    """The reviewed carrier a BOM row describes, or None."""
    from kicraft.design.part_identity import physical_inventory_record, reviewed_part

    record = physical_inventory_record(
        mpn=row.get("mpn"), symbol=row.get("symbol"), footprint=row.get("footprint")
    )
    if record is None:
        bundle = str(row.get("symbol") or "").partition(":")[0]
        if bundle:
            record = reviewed_part(bundle)
    return record


def _retail_and_catalog_stock(cid: str, _record: object) -> tuple[bool, str]:
    """Whether a C# can be bought now, and the plain-words reading when it cannot.

    The same two inventories §9.26 uses: the offline JLC catalog's floor and the live retail
    storefront. An unreachable storefront is not a shortfall (checks fail open).
    """
    from kicraft.parts_library import jlcparts, lcsc_retail

    key = str(cid or "").strip().upper()
    hit = jlcparts.lookup(key) if key else None
    floor = 100
    if hit is not None and (hit.get("stock") or 0) < floor:
        return False, (
            f"in the LCSC catalog with only {hit.get('stock') or 0} in stock for JLCPCB "
            f"assembly (< {floor})"
        )
    if not lcsc_retail.enabled():
        return True, ""
    info = lcsc_retail.stock(key)
    needed = max(int(info.get("min_buy") or 1), lcsc_retail.retail_floor())
    if int(info.get("stock") or 0) < needed:
        return False, (
            f"out of stock at the lcsc.com retail storefront ({info.get('stock')} available, "
            f"min buy {info.get('min_buy')})"
        )
    return True, ""


def decide(
    *,
    identity: str,
    mpn: str | None,
    family: str,
    shortfall: str,
    named_parts: Iterable[str] = (),
    brief: str = "",
    in_stock: Callable[[object], bool] = lambda _part: True,
) -> StockDecision:
    """The keep-or-swap decision for one part, with the sentence that explains it.

    ``shortfall`` is the reading in plain words ("out of stock at the lcsc.com retail storefront
    (0 available, min buy 5)") and lands in the reason unchanged, so the record says what was
    measured rather than that something went wrong.
    """
    named = part_number_named(identity, mpn, named_parts, brief)
    if named:
        return StockDecision(
            action=KEEP,
            named_in_brief=True,
            asked=True,
            alternative=None,
            reason=(
                f"{identity} ({mpn or identity}) is {shortfall}; the brief names this part, so it is "
                "kept — source it separately or say the word and it becomes an in-stock variation"
            ),
        )
    alternatives = swap_candidates(family, identity, in_stock=in_stock)
    if alternatives:
        chosen = alternatives[0]
        return StockDecision(
            action=SWAP,
            named_in_brief=False,
            asked=False,
            alternative=chosen,
            reason=(
                f"{identity} ({mpn or identity}) is {shortfall}; the pipeline chose it for the "
                f"class the brief names, so {getattr(chosen, 'identity', '')} — the same package, "
                "in stock — takes its place (defaulted)"
            ),
        )
    variants = family_variants(family, identity)
    detail = (
        "; other reviewed variants of this family exist but in different packages "
        f"({', '.join(str(getattr(row, 'identity', '')) for row in variants)}), which changes the "
        "pins and is a design decision"
        if variants
        else "; the library holds no other reviewed variant of this family"
    )
    return StockDecision(
        action=KEEP,
        named_in_brief=False,
        asked=False,
        alternative=None,
        reason=(
            f"{identity} ({mpn or identity}) is {shortfall} and the pipeline chose it for a class "
            f"the brief names, but no in-stock carrier shares its package{detail}"
        ),
    )


