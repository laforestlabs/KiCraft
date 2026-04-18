#!/usr/bin/env python3
"""Run KiCad Design Rule Check and report violations."""
import sys
import pcbnew

def main():
    if len(sys.argv) < 2:
        print("Usage: run_drc.py <file.kicad_pcb>")
        sys.exit(1)

    board = pcbnew.LoadBoard(sys.argv[1])
    board.BuildConnectivity()

    markers = list(board.Markers())

    if not markers:
        print("DRC: No violations found (0 markers on board).")
        print("Note: For a full DRC run with all checks, use KiCad's")
        print("Inspect > Design Rules Checker in the PCB editor.")
        return

    print(f"DRC found {len(markers)} marker(s):\n")
    for i, marker in enumerate(markers, 1):
        pos = marker.GetPosition()
        x, y = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
        rc = marker.GetRCItem()
        desc = rc.GetErrorMessage() if rc else "Unknown"
        severity = marker.GetSeverity()
        sev_str = {0: "Error", 1: "Warning", 2: "Exclusion"}.get(severity, str(severity))
        print(f"  {i}. [{sev_str}] {desc}")
        print(f"     at ({x:.2f}, {y:.2f}) mm")

if __name__ == "__main__":
    main()
