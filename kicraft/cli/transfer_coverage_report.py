#!/usr/bin/env python3
"""Report reviewed source-to-load power-transfer coverage (§9.39's input).

`§9.39 reviewed source-to-load power transfer` refuses a design whose conversion
cannot be proven, and what it proves it from is a graph: every reviewed
`power_transfer` fact becomes an edge only when both of its pin names resolve to
exactly ONE pin of the part's loaded symbol, both ends are wired to different
nets, and the part's MPN is an exact reviewed identity in the BOM
(`kicraft.design.synthesis.validation._reviewed_transfer_edges`). A series power
inductor (`L`) is the one non-reviewed witness the gate accepts.

A record can therefore *look* covered and still be inert, in three ways:

  * no `power_transfer` at all, on a family that is a power path;
  * a transfer whose pin name matches no pin, or two pins, of the symbol
    (an ambiguous `SW`/`GND` name can never resolve);
  * an identity-only record (`bundle=None`), which no BOM may select.

And a recipe can build a converter whose reviewed evidence is absent entirely,
which is the same hole seen from the other end: the part the recipe emits has no
record, so the graph has no edge for it.

This report joins those axes against the reviewed library and the shipped power
recipes, so the class of hole is visible without a live run:

    transfer-coverage-report
    transfer-coverage-report --json
"""

from __future__ import annotations

import argparse
import json
import sys

from kicraft.design.models import BOM, BomPart
from kicraft.design.part_identity import REVIEWED_PARTS, reviewed_part
from kicraft.design.recipes.wave_b_power import WAVE_B_POWER_RECIPES
from kicraft.design.synthesis.validation import (
    _fact_pin_name,
    _pin_info_by_ref,
    _pin_number_named,
)

# Family tokens that make a reviewed part a candidate power path. A part in one of
# these families is expected to say how energy crosses it; one that does not is the
# "no fact at all" hole. A record that already carries a fact is always listed,
# whatever its family, so a dead fact can never hide behind its family name.
_POWER_FAMILY_TOKENS = (
    "buck",
    "boost",
    "regulator",
    "converter",
    "charger",
    "pd-trigger",
    "pd-controller",
    "power-switch",
    "power-path",
    "ldo",
)

# Recipe parts whose role is a power path: the component that does the conversion.
_RECIPE_POWER_ROLES = frozenset({"converter", "regulator", "charger", "controller"})


def _transfer_paths(record) -> list[dict]:
    """``[{from_name, to_name, from_pin, to_pin}]`` resolved against its own symbol.

    Resolution uses the gate's own helpers, so a row this report calls
    edge-producible is exactly one `_reviewed_transfer_edges` would add.
    """
    info: dict = {}
    if record.symbol:
        bom = BOM(
            parts=[
                BomPart(
                    ref="U1",
                    value=record.identity,
                    symbol=record.symbol,
                    footprint=record.footprint or "unavailable:unavailable",
                    sheet="COVERAGE",
                    mpn=record.identity,
                )
            ]
        )
        info, _ = _pin_info_by_ref(bom)
    transfer = record.power_transfer or {}
    raw_paths = transfer.get("paths")
    if not isinstance(raw_paths, (list, tuple)):
        raw_paths = (transfer,)
    rows: list[dict] = []
    for path in raw_paths:
        if not isinstance(path, dict):
            continue
        source = str(path.get("from_pin") or "").upper() or _fact_pin_name(vars(record), "vin")
        dest = str(path.get("to_pin") or "").upper() or _fact_pin_name(vars(record), "ph")
        rows.append(
            {
                "from_name": source or None,
                "to_name": dest or None,
                "from_pin": _pin_number_named(info, "U1", source) if source else None,
                "to_pin": _pin_number_named(info, "U1", dest) if dest else None,
            }
        )
    return rows


def _record_row(record) -> dict:
    paths = _transfer_paths(record) if record.power_transfer else []
    if not record.power_transfer:
        status = "no-fact"
        reason = "family is a power path but the record carries no power_transfer fact"
    elif not record.symbol:
        status = "dead"
        reason = "identity-only record: no vendored symbol to resolve a pin against"
    elif any(row["from_pin"] and row["to_pin"] for row in paths):
        status = "edge-producible"
        reason = ""
    else:
        status = "dead"
        reason = "no path resolves both ends to exactly one pin of the loaded symbol"
    return {
        "identity": record.identity,
        "family": record.family,
        "bundle": record.bundle,
        "lcsc": record.lcsc,
        "transfer": dict(record.power_transfer),
        "paths": paths,
        "status": status,
        "reason": reason,
    }


def summarize() -> dict:
    records: list[dict] = []
    for record in REVIEWED_PARTS:
        if not record.power_transfer and not any(
            token in record.family for token in _POWER_FAMILY_TOKENS
        ):
            continue
        records.append(_record_row(record))

    recipes: list[dict] = []
    for recipe in WAVE_B_POWER_RECIPES:
        for group in recipe.parts:
            if group.role not in _RECIPE_POWER_ROLES:
                continue
            part = str(group.mpn or group.value or "")
            record = reviewed_part(part)
            recipes.append(
                {
                    "recipe": recipe.recipe,
                    "role": group.role,
                    "part": part,
                    "reviewed": record is not None,
                    "transfer": bool(record is not None and record.power_transfer),
                }
            )

    dead = [row for row in records if row["status"] == "dead"]
    no_fact = [row for row in records if row["status"] == "no-fact"]
    unwitnessed = [
        row for row in recipes if not row["reviewed"] or not row["transfer"]
    ]
    return {
        "records": records,
        "dead": dead,
        "no_fact": no_fact,
        "recipes": recipes,
        "unwitnessed_recipe_parts": unwitnessed,
    }


def format_report(s: dict) -> str:
    out: list[str] = []
    out.append("=" * 78)
    out.append(
        f"  KiCraft reviewed power-transfer coverage  "
        f"({len(s['records'])} records, {len(s['recipes'])} recipe power parts)"
    )
    out.append("=" * 78)
    out.append("  REVIEWED RECORDS  (status: from->to pin resolution against the symbol)")
    for row in sorted(s["records"], key=lambda r: (r["status"] != "edge-producible", r["identity"])):
        mark = {"edge-producible": "[ok  ]", "dead": "[DEAD]", "no-fact": "[NONE]"}[row["status"]]
        paths = ", ".join(
            f"{p['from_name'] or '?'}({p['from_pin'] or '-'})->{p['to_name'] or '?'}"
            f"({p['to_pin'] or '-'})"
            for p in row["paths"]
        )
        out.append(
            f"    {mark} {row['identity']:<26} {row['family']:<28} "
            f"{row['lcsc'] or '-':<9} {paths or '-'}"
        )
    out.append("  " + "-" * 74)
    out.append("  DEAD FACTS  (a reviewed transfer that can never become a graph edge):")
    if s["dead"]:
        for row in s["dead"]:
            out.append(f"    {row['identity']:<26} {row['reason']}")
    else:
        out.append("    (none)")
    out.append("  " + "-" * 74)
    out.append("  NO FACT  (a power family with no transfer at all):")
    if s["no_fact"]:
        for row in s["no_fact"]:
            out.append(f"    {row['identity']:<26} {row['family']:<28} {row['lcsc'] or '-'}")
    else:
        out.append("    (none)")
    out.append("  " + "-" * 74)
    out.append("  RECIPE POWER PARTS WITHOUT A REVIEWED TRANSFER  (§9.39 has no edge for them):")
    if s["unwitnessed_recipe_parts"]:
        for row in s["unwitnessed_recipe_parts"]:
            why = "no reviewed record" if not row["reviewed"] else "reviewed, no transfer fact"
            out.append(f"    {row['recipe']:<26} {row['part']:<20} {why}")
    else:
        out.append("    (none)")
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Report reviewed power-transfer coverage: which reviewed records can "
        "prove a source-to-load path for §9.39, which facts are dead, and which "
        "recipe-emitted converter parts have no reviewed transfer at all."
    )
    ap.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = ap.parse_args(argv)
    summary = summarize()
    if args.json:
        print(json.dumps(summary, indent=1, sort_keys=True))
    else:
        print(format_report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
