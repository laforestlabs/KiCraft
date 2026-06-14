#!/usr/bin/env python3
"""Construct the PARENT_LOCAL_CONN corpus fixture (construction record).

The corpus had no board exercising ``_snap_parent_local``'s connector branch:
in every workspace the edge connectors live INSIDE leaf subcircuits, so the
parent-local allowlist is empty and that branch never fires. Lever 2.1 (Phase 3
of docs/plans/place-route-root-cause-v2.md) deletes that branch, so it needs a
validating fixture.

This derives one from the proven-good USB_PD_TRIGGER fixture by hand (no LLM
synthesis -- deterministic, free), changing as little as possible:

  * rename the project USB_PD_TRIGGER -> PARENT_LOCAL_CONN (dir == stem == config
    stem, so discover_project_config + the replay-corpus golden line up);
  * clone the small R3 (0805) footprint into J3 at root level -- in NO leaf, so
    compose extracts it as a PARENT-LOCAL connector (ref "J" + edge zone ->
    connector branch). Both pads forced to GND so J3's stamped copper can never
    short against the leaves' GND;
  * add ``J3: {edge: bottom}`` to component_zones (J1/J2/SW1 stay edge-zoned
    leaf connectors -- the positive control);
  * record ``parent_compose_spacing_mm: 3.5``: a 4th edge connector packs the
    already-dense board tight enough to short at the 2.0 default, so the gate
    composes this fixture at the clearance it was frozen with.

The committed fixture is the source of truth; re-running this mints fresh UUIDs
(placement is UUID-independent, so the golden is unaffected -- but regenerate it
with ``scripts/replay_corpus.py --mode parent --update`` after any re-run).

Usage:  python scripts/build_parent_local_conn_fixture.py
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "tests" / "fixtures" / "replay_workspace"
SRC = ROOT / "USB_PD_TRIGGER"
DST = ROOT / "PARENT_LOCAL_CONN"


def main() -> None:
    shutil.rmtree(DST, ignore_errors=True)
    shutil.copytree(SRC, DST)

    # 1. rename project files + internal stem references
    for ext in ("kicad_pcb", "kicad_pro", "kicad_sch"):
        (DST / f"USB_PD_TRIGGER.{ext}").rename(DST / f"PARENT_LOCAL_CONN.{ext}")
    (DST / "USB_PD_TRIGGER_autoplacer.json").rename(
        DST / "PARENT_LOCAL_CONN_autoplacer.json")
    pro = DST / "PARENT_LOCAL_CONN.kicad_pro"
    pro.write_text(pro.read_text().replace(
        "USB_PD_TRIGGER.kicad_pro", "PARENT_LOCAL_CONN.kicad_pro"))
    sch = DST / "PARENT_LOCAL_CONN.kicad_sch"
    sch.write_text(sch.read_text().replace(
        '(project "USB_PD_TRIGGER"', '(project "PARENT_LOCAL_CONN"'))

    # 2. clone R3 (0805) -> J3 at root level, both pads GND, repositioned
    pcb = DST / "PARENT_LOCAL_CONN.kicad_pcb"
    lines = pcb.read_text().split("\n")
    start = next(i for i, ln in enumerate(lines)
                 if ln.strip().startswith("(footprint ")
                 and any('"Reference" "R3"' in lines[j] for j in range(i, i + 25)))
    end = next(k for k in range(start + 1, len(lines)) if lines[k] == "\t)")
    block = "\n".join(lines[start:end + 1])
    gnd = re.search(r'\(net \d+ "GND"\)', block)
    if gnd is None:
        raise SystemExit("R3 has no GND pad to copy onto J3")
    block = block.replace('(property "Reference" "R3"',
                          '(property "Reference" "J3"', 1)
    block = re.sub(r'\(uuid "[0-9a-fA-F-]+"\)',
                   lambda _m: f'(uuid "{uuid.uuid4()}")', block)
    block = re.sub(r'\n\t\t\(at [0-9.\- ]+\)', '\n\t\t(at 150.0 122.0)',
                   block, count=1)
    block = re.sub(r'\(net \d+ "[^"]*"\)', gnd.group(0), block)  # both pads -> GND
    lines = lines[:end + 1] + block.split("\n") + lines[end + 1:]
    pcb.write_text("\n".join(lines))

    # 3. config: rename + add the parent-local J3 zone + per-fixture spacing
    cfg = DST / "PARENT_LOCAL_CONN_autoplacer.json"
    d = json.loads(cfg.read_text())
    d["project_name"] = "PARENT_LOCAL_CONN"
    d["pcb_file"] = "PARENT_LOCAL_CONN.kicad_pcb"
    d["component_zones"]["J3"] = {"edge": "bottom"}
    d["parent_compose_spacing_mm"] = 3.5
    cfg.write_text(json.dumps(d, indent=2) + "\n")

    print(f"built {DST.relative_to(REPO)}")
    print("next: python scripts/replay_corpus.py --mode parent --update")


if __name__ == "__main__":
    main()
