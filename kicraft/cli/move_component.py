#!/usr/bin/env python3
"""Move a footprint to an absolute position."""
import sys
import argparse
import pcbnew

def main():
    parser = argparse.ArgumentParser(description="Move a component to absolute position")
    parser.add_argument("pcb", help="Path to .kicad_pcb file")
    parser.add_argument("ref", help="Component reference (e.g. U1, R3)")
    parser.add_argument("x", type=float, help="Target X position in mm")
    parser.add_argument("y", type=float, help="Target Y position in mm")
    parser.add_argument("--rotate-deg", type=float, default=None, help="Set orientation in degrees")
    parser.add_argument("--in-place", action="store_true", help="Overwrite original file")
    args = parser.parse_args()

    board = pcbnew.LoadBoard(args.pcb)
    fp = board.FindFootprintByReference(args.ref)
    if not fp:
        print(f"Error: footprint '{args.ref}' not found")
        sys.exit(1)

    old_pos = fp.GetPosition()
    print(f"Moving {args.ref} from ({pcbnew.ToMM(old_pos.x):.2f}, {pcbnew.ToMM(old_pos.y):.2f})")

    new_pos = pcbnew.VECTOR2I(pcbnew.FromMM(args.x), pcbnew.FromMM(args.y))
    fp.SetPosition(new_pos)
    print(f"  to ({args.x:.2f}, {args.y:.2f})")

    if args.rotate_deg is not None:
        fp.SetOrientationDegrees(args.rotate_deg)
        print(f"  rotation: {args.rotate_deg}°")

    out = args.pcb if args.in_place else args.pcb.replace(".kicad_pcb", "_modified.kicad_pcb")
    board.Save(out)
    print(f"Saved to: {out}")

if __name__ == "__main__":
    main()
