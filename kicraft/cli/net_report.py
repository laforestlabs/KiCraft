#!/usr/bin/env python3
"""List all nets with pad counts."""
import sys
import pcbnew

def main():
    if len(sys.argv) < 2:
        print("Usage: net_report.py <file.kicad_pcb>")
        sys.exit(1)

    board = pcbnew.LoadBoard(sys.argv[1])
    board.BuildConnectivity()

    board.GetNetsByName()
    net_pads = {}

    for pad in board.GetPads():
        name = pad.GetNetname()
        if name not in net_pads:
            net_pads[name] = 0
        net_pads[name] += 1

    print(f"{'Net':<30} {'Pads':>5}")
    print("-" * 37)
    for name in sorted(net_pads.keys()):
        if name == "":
            continue
        print(f"{name:<30} {net_pads[name]:>5}")

    print(f"\nTotal nets: {len(net_pads) - (1 if '' in net_pads else 0)}")

if __name__ == "__main__":
    main()
