#!/usr/bin/env python3
"""Add a GND copper pour zone on B.Cu covering the full board."""
import argparse
import pcbnew


def add_gnd_zone(pcb_path, in_place=False):
    board = pcbnew.LoadBoard(pcb_path)

    # Get board outline
    rect = board.GetBoardEdgesBoundingBox()
    x1 = rect.GetX()
    y1 = rect.GetY()
    x2 = x1 + rect.GetWidth()
    y2 = y1 + rect.GetHeight()

    # Inset slightly from edge
    margin = pcbnew.FromMM(0.5)
    x1 += margin
    y1 += margin
    x2 -= margin
    y2 -= margin

    # Find GND net
    gnd_net = board.GetNetInfo().GetNetItem("GND")
    if not gnd_net or gnd_net.GetNetCode() == 0:
        print("ERROR: GND net not found")
        return

    # Create zone on B.Cu
    zone = pcbnew.ZONE(board)
    zone.SetNet(gnd_net)
    zone.SetLayer(pcbnew.B_Cu)
    zone.SetIsRuleArea(False)
    zone.SetDoNotAllowTracks(False)
    zone.SetDoNotAllowVias(False)
    zone.SetDoNotAllowPads(False)
    zone.SetDoNotAllowCopperPour(False)

    # Set zone properties
    zone.SetLocalClearance(pcbnew.FromMM(0.3))
    zone.SetMinThickness(pcbnew.FromMM(0.25))
    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
    zone.SetThermalReliefGap(pcbnew.FromMM(0.5))
    zone.SetThermalReliefSpokeWidth(pcbnew.FromMM(0.5))
    zone.SetAssignedPriority(0)

    # Create outline
    outline = zone.Outline()
    outline.NewOutline()
    outline.Append(x1, y1)
    outline.Append(x2, y1)
    outline.Append(x2, y2)
    outline.Append(x1, y2)

    board.Add(zone)

    # Fill zones
    filler = pcbnew.ZONE_FILLER(board)
    filler.Fill(board.Zones())

    out = pcb_path if in_place else pcb_path.replace(".kicad_pcb", "_gnd.kicad_pcb")
    board.Save(out)
    print(f"Added GND zone on B.Cu, saved to: {out}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("pcb")
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    add_gnd_zone(args.pcb, args.in_place)


if __name__ == "__main__":
    main()
