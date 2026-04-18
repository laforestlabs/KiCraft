#!/usr/bin/env python3
"""Arrange footprints matching a reference prefix into a grid."""
import sys
import argparse
import re
import pcbnew

def natural_sort_key(ref):
    """Sort D1, D2, D10 correctly."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', ref)]

def main():
    parser = argparse.ArgumentParser(description="Arrange footprints in a grid")
    parser.add_argument("pcb", help="Path to .kicad_pcb file")
    parser.add_argument("prefix", help="Reference prefix to match (e.g. D, LED, R)")
    parser.add_argument("--cols", type=int, required=True, help="Number of columns")
    parser.add_argument("--spacing-mm", type=float, required=True, help="Grid spacing in mm")
    parser.add_argument("--start-x", type=float, default=100.0, help="Grid origin X in mm")
    parser.add_argument("--start-y", type=float, default=100.0, help="Grid origin Y in mm")
    parser.add_argument("--in-place", action="store_true", help="Overwrite original file")
    args = parser.parse_args()

    board = pcbnew.LoadBoard(args.pcb)
    pattern = re.compile(rf'^{re.escape(args.prefix)}\d+$')

    fps = []
    for fp in board.Footprints():
        ref = fp.GetReferenceAsString()
        if pattern.match(ref):
            fps.append(fp)

    fps.sort(key=lambda f: natural_sort_key(f.GetReferenceAsString()))

    if not fps:
        print(f"No footprints matching '{args.prefix}*' found.")
        sys.exit(1)

    print(f"Arranging {len(fps)} footprints in {args.cols}-column grid:")
    for i, fp in enumerate(fps):
        col = i % args.cols
        row = i // args.cols
        x = args.start_x + col * args.spacing_mm
        y = args.start_y + row * args.spacing_mm
        fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y)))
        print(f"  {fp.GetReferenceAsString()} -> ({x:.2f}, {y:.2f})")

    out = args.pcb if args.in_place else args.pcb.replace(".kicad_pcb", "_modified.kicad_pcb")
    board.Save(out)
    print(f"\nSaved to: {out}")

if __name__ == "__main__":
    main()
