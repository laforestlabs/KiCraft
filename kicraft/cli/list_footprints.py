#!/usr/bin/env python3
"""List all footprints in a KiCad PCB with reference, value, position, and layer."""

import argparse
import sys

import pcbnew


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="List all footprints in a KiCad PCB file."
    )
    parser.add_argument("pcb", help="Path to .kicad_pcb board file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    board = pcbnew.LoadBoard(args.pcb)
    if board is None:
        print("error: failed to load board: " + args.pcb, file=sys.stderr)
        sys.exit(1)

    fps = list(board.Footprints())
    fps.sort(key=lambda f: f.GetReferenceAsString())

    header = "{:<8} {:<20} {:>8} {:>8} {:>6} {:<8}".format(
        "Ref", "Value", "X(mm)", "Y(mm)", "Rot", "Layer"
    )
    print(header)
    print("-" * 70)
    for fp in fps:
        ref = fp.GetReferenceAsString()
        val = fp.GetFieldText("Value")
        pos = fp.GetPosition()
        x, y = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
        rot = fp.GetOrientationDegrees()
        layer = "Front" if fp.GetLayer() == pcbnew.F_Cu else "Back"
        print("{:<8} {:<20} {:>8.2f} {:>8.2f} {:>6.1f} {:<8}".format(
            ref, val, x, y, rot, layer
        ))

    print()
    print("Total: {} footprints".format(len(fps)))


if __name__ == "__main__":
    main()
