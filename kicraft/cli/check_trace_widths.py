#!/usr/bin/env python3
"""Find traces narrower than a minimum width."""
import sys
import argparse
import pcbnew

def main():
    parser = argparse.ArgumentParser(description="Check trace widths against minimum")
    parser.add_argument("pcb", help="Path to .kicad_pcb file")
    parser.add_argument("--min-mm", type=float, default=0.127, help="Minimum trace width in mm (default: 0.127)")
    args = parser.parse_args()

    board = pcbnew.LoadBoard(args.pcb)
    min_nm = pcbnew.FromMM(args.min_mm)
    violations = []

    for track in board.GetTracks():
        if isinstance(track, pcbnew.PCB_VIA):
            continue
        w = track.GetWidth()
        if w < min_nm:
            start = track.GetStart()
            violations.append({
                "width_mm": pcbnew.ToMM(w),
                "net": track.GetNetname(),
                "layer": track.GetLayerName(),
                "x": pcbnew.ToMM(start.x),
                "y": pcbnew.ToMM(start.y),
            })

    if violations:
        print(f"Found {len(violations)} traces narrower than {args.min_mm}mm:\n")
        print(f"{'Width(mm)':>10} {'Net':<20} {'Layer':<10} {'X':>8} {'Y':>8}")
        print("-" * 60)
        for v in violations:
            print(f"{v['width_mm']:>10.3f} {v['net']:<20} {v['layer']:<10} {v['x']:>8.2f} {v['y']:>8.2f}")
    else:
        print(f"All traces are >= {args.min_mm}mm. No violations found.")

if __name__ == "__main__":
    main()
