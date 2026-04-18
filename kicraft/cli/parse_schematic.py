#!/usr/bin/env python3
"""Parse KiCad schematic to extract component groupings.

Extracts groups from schematic text labels and component associations.
"""
import argparse
import json
import re
import sys
from collections import defaultdict


# Legacy fallback: LLUPS-specific IC groups.
# Prefer loading from a project config JSON via --config instead.
_LEGACY_IC_GROUPS = {
    "U1": ["C1", "R1", "R2", "F1", "J1"],  # USB-C controller
    "U2": ["C2", "C3", "C4", "R3", "R4", "R5", "R6", "R7", "R8", "RT1", "D1", "D2"],  # BQ24072 charger
    "U3": ["Q1", "U6"],  # HY2113 protection + LN61C supervisor
    "U4": ["C5", "C6", "C7", "L1", "D3"],  # MT3608 boost
    "U5": ["C8", "R9", "R10", "R11", "J2", "J3"],  # AP2112 LDO + output/debug connectors
    "BT1": ["BT2"],  # battery cells
}


def load_ic_groups(config_path=None):
    """Load IC group definitions from a project config JSON, or fall back to legacy defaults.

    Args:
        config_path: Optional path to a project config JSON containing an "ic_groups" key.

    Returns:
        dict mapping IC reference to list of supporting component references.
    """
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
        if "ic_groups" in cfg:
            return cfg["ic_groups"]
        print(f"Warning: config {config_path} has no 'ic_groups' key; using legacy fallback",
              file=sys.stderr)
    return dict(_LEGACY_IC_GROUPS)


# Module-level alias kept for any external importers (populated at parse time
# or on demand via load_ic_groups).
IC_GROUPS = _LEGACY_IC_GROUPS


def parse_schematic_groups(sch_path):
    """Parse schematic and return component groups based on symbols near labels."""
    with open(sch_path) as f:
        txt = f.read()

    # Known group labels from schematic
    group_labels = {
        "USB INPUT": (25, 40),
        "CHARGER (BQ24072)": (125, 40),
        "BATTERY + PROTECTION": (240, 40),
        "BOOST 5V (MT3608)": (125, 185),
        "LDO 3.3V + OUTPUT": (250, 185),
    }

    # Extract components by reference
    comp_refs = set()
    for m in re.finditer(r'\(property "Reference" "([A-Z][0-9]+)"', txt):
        comp_refs.add(m.group(1))

    # Build group->components mapping
    groups = defaultdict(list)

    # Map ICs to their group based on label position
    for m in re.finditer(r'\(property "Reference" "(U[0-9]+)".*?\n.*?\(at ([-\d.]+) ([-\d.]+)', txt, re.DOTALL):
        ref = m.group(1)
        x, y = float(m.group(2)), float(m.group(3))

        # Find nearest label
        for label, (lx, ly) in group_labels.items():
            dist = ((x - lx)**2 + (y - ly)**2)**0.5
            if dist < 100:  # within reasonable distance
                groups[label].append(ref)
                break

    print("=== Schematic Groups ===")
    for label, comps in sorted(groups.items()):
        print(f"  {label}: {sorted(comps)}")

    return groups, comp_refs


def build_schematic_groups(pcb_nets):
    """Build groups from PCB net data - components sharing nets with ICs."""
    from collections import defaultdict

    # Find which IC each component connects to
    ic_connections = defaultdict(set)

    for net_name, net in pcb_nets.items():
        if net_name in ("GND", "/GND"):
            continue
        refs = net.component_refs
        for ref in refs:
            if ref.startswith("U") or ref.startswith("IC"):
                ic_connections[ref].update(refs)

    return ic_connections



def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Parse KiCad schematic to extract component groupings.")
    parser.add_argument("schematic", help="Path to .kicad_sch file")
    parser.add_argument("--config", default=None,
                        help="Path to project config JSON (e.g. LLUPS_autoplacer.json) "
                             "containing an 'ic_groups' key")
    args = parser.parse_args()

    load_ic_groups(args.config)
    parse_schematic_groups(args.schematic)


if __name__ == "__main__":
    main()
