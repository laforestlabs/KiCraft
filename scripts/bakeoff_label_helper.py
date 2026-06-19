#!/usr/bin/env python3
"""§9 labeling helper for the electrical-review model bakeoff.

Loads each ``state.json``, reconstructs the BOM, and runs the five deterministic
semantic gates the synthesis stage runs unconditionally:

    §9.16 power pin polarity      check_power_pin_polarity
    §9.17 two-terminal self-short check_two_terminal_self_short
    §9.18 rf feed isolation       check_rf_feed_isolation
    §9.19 single net per pin      check_single_net_per_pin
    §9.20 family wiring contracts check_family_wiring_contracts

These gates RAISE during synthesis (validation.run_validations), so every
*completed* board in a self-eval snapshot already passed them. The point of this
helper is to:

  (a) confirm each synthetic injection trips the gate it targets, and
  (b) confirm the natural corpus is §9-clean -- which is the proof that the
      LLM electrical-review gate addresses a defect class §9 cannot see.

Pin names resolve through the same ``project_root=None`` four-tier search the
production §9 checks use (see validation._pin_info_by_ref), so results match what
the gate sees in production. Run from the repo root.

Usage:
    python scripts/bakeoff_label_helper.py <state.json | dir> [...]
    python scripts/bakeoff_label_helper.py logs/bakeoff/<ts>/corpus
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from kicraft.design.cli_app import _load_state
from kicraft.design.synthesis.validation import (
    check_family_wiring_contracts,
    check_power_pin_polarity,
    check_rf_feed_isolation,
    check_single_net_per_pin,
    check_two_terminal_self_short,
)

CHECKS = (
    check_power_pin_polarity,       # §9.16
    check_two_terminal_self_short,  # §9.17
    check_rf_feed_isolation,        # §9.18
    check_single_net_per_pin,       # §9.19
    check_family_wiring_contracts,  # §9.20
)


def _design_name(state_path: Path) -> str:
    p = state_path.parent
    return p.parent.name if p.name == ".kicraft" else p.name


def run_one(state_path: Path) -> dict:
    name = _design_name(state_path)
    try:
        state = _load_state(state_path)
    except Exception as e:  # noqa: BLE001 - report, don't crash the sweep
        return {"design": name, "error": f"{type(e).__name__}: {e}"}
    bom = getattr(state, "bom", None)
    if bom is None or not getattr(bom, "connections", None):
        return {"design": name, "skipped": "no wired BOM"}
    fired = []
    for chk in CHECKS:
        res = chk(bom)
        if not res.ok:
            fired.append({
                "gate": res.name,
                "message": res.message,
                "offenders": list(res.offenders),
            })
    return {"design": name, "clean": not fired, "fired": fired}


def _iter_state_paths(arg: str):
    p = Path(arg)
    if p.is_file():
        yield p
    elif p.is_dir():
        # corpus/<id>/state.json  or  snapshot/run_*/.kicraft/state.json
        yield from sorted(p.glob("*/state.json"))
        yield from sorted(p.glob("*/.kicraft/state.json"))
    else:
        print(f"warning: no such path {arg}", file=sys.stderr)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    paths = []
    for a in argv:
        paths.extend(_iter_state_paths(a))
    results = [run_one(sp) for sp in paths]

    fired = [r for r in results if r.get("fired")]
    clean = [r for r in results if r.get("clean")]
    other = [r for r in results if "error" in r or "skipped" in r]

    for r in results:
        if r.get("fired"):
            gates = ", ".join(f["gate"] for f in r["fired"])
            print(f"FIRED  {r['design']:32s} {gates}")
            for f in r["fired"]:
                for off in f["offenders"]:
                    print(f"         - {off}")
        elif r.get("clean"):
            print(f"clean  {r['design']:32s} §9.16-§9.20 ok")
        else:
            print(f"skip   {r['design']:32s} {r.get('error') or r.get('skipped')}")

    print(f"\n{len(clean)} clean, {len(fired)} fired §9, {len(other)} skipped/error")
    print(json.dumps(results))  # machine-readable tail
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
