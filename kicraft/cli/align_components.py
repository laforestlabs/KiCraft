#!/usr/bin/env python3
"""Align footprints along an axis (match X or Y coordinate)."""
import sys
import argparse
import pcbnew

def main():
    parser = argparse.ArgumentParser(description="Align components along an axis")
    parser.add_argument("pcb", help="Path to .kicad_pcb file")
    parser.add_argument("refs", nargs="+", help="Component references to align (e.g. R1 R2 R3)")
    parser.add_argument("--axis", choices=["x", "y"], required=True,
                        help="Axis to align along: 'x' = same X (vertical line), 'y' = same Y (horizontal line)")
    parser.add_argument("--to", type=float, default=None,
                        help="Align to this coordinate in mm (default: average of current positions)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite original file")
    args = parser.parse_args()

    board = pcbnew.LoadBoard(args.pcb)
    fps = []
    for ref in args.refs:
        fp = board.FindFootprintByReference(ref)
        if not fp:
            print(f"Warning: '{ref}' not found, skipping")
            continue
        fps.append(fp)

    if not fps:
        print("No valid footprints found.")
        sys.exit(1)

    if args.to is not None:
        target = args.to
    else:
        if args.axis == "x":
            target = sum(pcbnew.ToMM(fp.GetPosition().x) for fp in fps) / len(fps)
        else:
            target = sum(pcbnew.ToMM(fp.GetPosition().y) for fp in fps) / len(fps)

    print(f"Aligning {len(fps)} components to {args.axis}={target:.2f}mm:")
    for fp in fps:
        pos = fp.GetPosition()
        if args.axis == "x":
            new_pos = pcbnew.VECTOR2I(pcbnew.FromMM(target), pos.y)
        else:
            new_pos = pcbnew.VECTOR2I(pos.x, pcbnew.FromMM(target))
        fp.SetPosition(new_pos)
        print(f"  {fp.GetReferenceAsString()} -> ({pcbnew.ToMM(new_pos.x):.2f}, {pcbnew.ToMM(new_pos.y):.2f})")

    out = args.pcb if args.in_place else args.pcb.replace(".kicad_pcb", "_modified.kicad_pcb")
    board.Save(out)
    print(f"\nSaved to: {out}")

if __name__ == "__main__":
    main()
