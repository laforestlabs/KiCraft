"""What to do about a part that is real but out of stock.

The owner's rule (2026-09-26): a board must still get made when one or two of its parts cannot be
bought today. Three outcomes, decided here so every caller decides them the same way:

- **keep** a part the brief names by its own order code. The owner may hold stock or have a second
  source, and only they can weigh that. A person is asked when one is watching; an auto-defaulted
  run keeps the part and records the shortfall so the board is not held up by it.
- **swap** a part the pipeline chose for a class the brief named only when a reviewed equivalent
  device in the same package is in stock. Sharing a family and package alone does not establish
  compatible ratings, values or pin functions.
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

    This checks physical compatibility only; callers must separately prove device equivalence.
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

    A common family and footprint do not prove interchangeable devices. Only explicit reviewed
    order-code equivalence or a record's reviewed same-device/package/grade relation is accepted.
    """
    from kicraft.design.part_identity import (
        reviewed_identity_for_order_code, reviewed_inventory, reviewed_part,
    )

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
        current_id = str(getattr(current, "identity", "")).strip().casefold()
        candidate_id = str(getattr(candidate, "identity", "")).strip().casefold()
        current_limits = getattr(current, "operating_limits", {}) or {}
        candidate_limits = getattr(candidate, "operating_limits", {}) or {}
        equivalent = (
            reviewed_identity_for_order_code(current_id) == reviewed_identity_for_order_code(candidate_id)
            or str(current_limits.get("same_device_package_grade_as", "")).casefold() == candidate_id
            or str(candidate_limits.get("same_device_package_grade_as", "")).casefold() == current_id
        )
        if not equivalent:
            continue
        if in_stock(candidate):
            out.append(candidate)
    return tuple(sorted(out, key=lambda row: str(getattr(row, "identity", ""))))


def family_variants(family: str, identity: str) -> tuple[object, ...]:
    """Other reviewed carriers of the family in a different package."""
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
                and (current is None or not same_package(current, row))
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
        # Only explicitly equivalent devices can replace this part without a design change.
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
        else "; the library holds no other reviewed variant in a different package"
    )
    return StockDecision(
        action=KEEP,
        named_in_brief=False,
        asked=False,
        alternative=None,
        reason=(
            f"{identity} ({mpn or identity}) is {shortfall} and the pipeline chose it for a class "
            f"the brief names, but no in-stock reviewed equivalent shares its package{detail}"
        ),
    )


def validate_bom_accessories(bom, architecture) -> tuple[list[str], list[dict]]:
    """Qualify non-PCB purchases and retain the lowerer's required accessories.

    Quantities are per owner, and catalog identity must match exactly: a shunt
    cannot be inferred from the header's MPN or silently disappear from its BOM.
    """
    from kicraft.design.lowering import lower_requirement
    from kicraft.parts_library import jlcparts

    requirements = {row.id: row for row in (getattr(architecture, "requirements", ()) or ())}
    errors: list[str] = []
    totals: dict[tuple[str, str], int] = {}
    owners: dict[tuple[str, str], list[dict]] = {}
    for requirement in requirements.values():
        try:
            artifact = lower_requirement(requirement)
        except (TypeError, ValueError):
            continue  # Other construction diagnostics own an unlowerable requirement.
        if artifact is None:
            continue
        for group in artifact.groups:
            if not group.accessories:
                continue
            matches = [
                part for part in bom.parts
                if part.resolution_source == "lowerer"
                and part.resolution_id == artifact.lowerer_id
                and part.lowering_requirement_id == requirement.id
                and part.lowering_role == group.role
                and part.mpn == group.mpn
                and part.symbol == group.symbol
                and part.footprint == group.footprint
            ]
            if len(matches) != group.quantity:
                errors.append(
                    f"{requirement.id}: required accessory-bearing {group.role} owners "
                    f"expected {group.quantity}, found {len(matches)}"
                )
    for part in bom.parts:
        accessories = tuple(part.accessories)
        if part.resolution_source == "lowerer" and part.lowering_requirement_id in requirements:
            try:
                artifact = lower_requirement(requirements[part.lowering_requirement_id])
            except (TypeError, ValueError) as exc:
                errors.append(f"{part.ref}: cannot verify accessory ownership: {exc}")
                continue
            if artifact is not None and artifact.lowerer_id == part.resolution_id:
                group = next((g for g in artifact.groups if g.role == part.lowering_role), None)
                if group is not None and tuple(group.accessories) != accessories:
                    errors.append(f"{part.ref}: required lowerer accessories do not match the BOM")
        for index, item in enumerate(accessories, 1):
            key = (item.lcsc.upper(), item.mpn.casefold())
            totals[key] = totals.get(key, 0) + item.quantity
            owners.setdefault(key, []).append({
                "accessory_id": f"{part.ref}:A{index}", "owner_ref": part.ref,
                "quantity": item.quantity,
            })
    receipt: list[dict] = []
    for (lcsc, mpn), quantity in totals.items():
        row = jlcparts.lookup(lcsc)
        if row is None or (row.get("model") or "").casefold() != mpn:
            errors.append(f"{lcsc} ({mpn}): accessory is absent or has a different catalog MPN")
            continue
        stock = row.get("stock") or 0
        if stock < quantity:
            errors.append(f"{lcsc} ({mpn}): accessory stock {stock} is below quantity {quantity}")
            continue
        receipt.append({
            "kind": "accessory", "exact_lcsc": lcsc, "catalog_mpn": row["model"],
            "quantity": quantity, "stock": stock, "owners": owners[(lcsc, mpn)],
        })
    return errors, receipt
