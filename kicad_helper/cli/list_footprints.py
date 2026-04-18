#!/usr/bin/env python3
"""List all footprints in a KiCad PCB with reference, value, position, and layer."""
import sys
import pcbnew

def main():
    if len(sys.argv) < 2:
        print("Usage: list_footprints.py <file.kicad_pcb>")
        sys.exit(1)

    board = pcbnew.LoadBoard(sys.argv[1])
    fps = list(board.Footprints())
    fps.sort(key=lambda f: f.GetReferenceAsString())

    print(f"{'Ref':<8} {'Value':<20} {'X(mm)':>8} {'Y(mm)':>8} {'Rot':>6} {'Layer':<8}")
    print("-" * 70)
    for fp in fps:
        ref = fp.GetReferenceAsString()
        val = fp.GetFieldText("Value")
        pos = fp.GetPosition()
        x, y = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
        rot = fp.GetOrientationDegrees()
        layer = "Front" if fp.GetLayer() == pcbnew.F_Cu else "Back"
        print(f"{ref:<8} {val:<20} {x:>8.2f} {y:>8.2f} {rot:>6.1f} {layer:<8}")

    print(f"\nTotal: {len(fps)} footprints")

if __name__ == "__main__":
    main()
