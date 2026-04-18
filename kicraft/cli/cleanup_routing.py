#!/usr/bin/env python3
"""Clean up routing issues: remove dangling vias, crossing tracks, and
tracks that create shorts between nets.

Usage:
    python3 cleanup_routing.py <pcb> [--remove-dangling] [--remove-crossings] [--dry-run]
"""
import argparse
import math
import pcbnew


def find_dangling_vias(board, thermal_refs=None, thermal_radius_mm=3.0):
    """Find vias that have no tracks connecting to them on any layer.
    Preserves vias near thermal_refs ICs (thermal vias for heat dissipation)."""
    thermal_refs = thermal_refs or []
    vias = []
    tracks = []
    for t in board.GetTracks():
        if isinstance(t, pcbnew.PCB_VIA):
            vias.append(t)
        else:
            tracks.append(t)

    # Build thermal exclusion zones
    thermal_zones = []
    for ref in thermal_refs:
        fp = board.FindFootprintByReference(ref)
        if fp:
            pos = fp.GetPosition()
            thermal_zones.append((pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)))

    dangling = []
    for via in vias:
        vpos = via.GetPosition()
        vx, vy = vpos.x, vpos.y
        vx_mm, vy_mm = pcbnew.ToMM(vx), pcbnew.ToMM(vy)

        # Skip vias near thermal ICs
        near_thermal = False
        for tx, ty in thermal_zones:
            if math.hypot(vx_mm - tx, vy_mm - ty) <= thermal_radius_mm:
                near_thermal = True
                break
        if near_thermal:
            continue

        connected = False
        for track in tracks:
            sx, sy = track.GetStart().x, track.GetStart().y
            ex, ey = track.GetEnd().x, track.GetEnd().y
            if (sx == vx and sy == vy) or (ex == vx and ey == vy):
                connected = True
                break
        if not connected:
            for pad in board.GetPads():
                ppos = pad.GetPosition()
                if ppos.x == vx and ppos.y == vy:
                    connected = True
                    break
        if not connected:
            dangling.append(via)
    return dangling


def find_crossing_tracks(board):
    """Find pairs of tracks from different nets that cross each other."""
    tracks = [t for t in board.GetTracks() if not isinstance(t, pcbnew.PCB_VIA)]

    # Group by layer
    by_layer = {}
    for t in tracks:
        layer = t.GetLayer()
        by_layer.setdefault(layer, []).append(t)

    crossings = []
    for layer, layer_tracks in by_layer.items():
        for i in range(len(layer_tracks)):
            for j in range(i + 1, len(layer_tracks)):
                a, b = layer_tracks[i], layer_tracks[j]
                if a.GetNetname() == b.GetNetname():
                    continue  # same net, not a problem
                if _segments_cross(a, b):
                    crossings.append((a, b))
    return crossings


def _segments_cross(a, b):
    """Check if two track segments actually cross (intersect, not just touch)."""
    ax1, ay1 = a.GetStart().x, a.GetStart().y
    ax2, ay2 = a.GetEnd().x, a.GetEnd().y
    bx1, by1 = b.GetStart().x, b.GetStart().y
    bx2, by2 = b.GetEnd().x, b.GetEnd().y

    d = (bx2 - bx1) * (ay2 - ay1) - (by2 - by1) * (ax2 - ax1)
    if d == 0:
        return False  # parallel

    ua = ((bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)) / d
    ub = ((ax2 - ax1) * (ay1 - by1) - (ay2 - ay1) * (ax1 - bx1)) / d

    # Strictly inside both segments (not at endpoints)
    return 0.01 < ua < 0.99 and 0.01 < ub < 0.99


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcb")
    parser.add_argument("--remove-dangling", action="store_true")
    parser.add_argument("--remove-crossings", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--thermal-refs", default=None,
                        help="Comma-separated component refs for thermal via exclusion zones "
                             "(e.g. U2,U4). Defaults to empty list if not provided.")
    args = parser.parse_args()

    # Parse thermal refs from CLI or default to empty
    thermal_refs = [r.strip() for r in args.thermal_refs.split(",") if r.strip()] if args.thermal_refs else []

    board = pcbnew.LoadBoard(args.pcb)

    # Dangling vias (preserve thermal vias near power ICs)
    dangling = find_dangling_vias(board, thermal_refs=thermal_refs, thermal_radius_mm=3.0)
    print(f"Dangling vias: {len(dangling)}")
    for v in dangling[:10]:
        pos = v.GetPosition()
        print(f"  ({pcbnew.ToMM(pos.x):.2f}, {pcbnew.ToMM(pos.y):.2f}) net={v.GetNetname()}")
    if len(dangling) > 10:
        print(f"  ... and {len(dangling) - 10} more")

    # Crossing tracks
    crossings = find_crossing_tracks(board)
    print(f"\nCrossing track pairs: {len(crossings)}")
    for a, b in crossings[:10]:
        print(f"  {a.GetNetname()} x {b.GetNetname()} on {a.GetLayerName()}")
    if len(crossings) > 10:
        print(f"  ... and {len(crossings) - 10} more")

    if args.dry_run:
        return

    removed = 0
    if args.remove_dangling:
        for v in dangling:
            board.Remove(v)
            removed += 1
        print(f"\nRemoved {len(dangling)} dangling vias")

    if args.remove_crossings:
        # Remove the shorter track in each crossing pair
        to_remove = set()
        for a, b in crossings:
            la = math.hypot(
                pcbnew.ToMM(a.GetEnd().x) - pcbnew.ToMM(a.GetStart().x),
                pcbnew.ToMM(a.GetEnd().y) - pcbnew.ToMM(a.GetStart().y))
            lb = math.hypot(
                pcbnew.ToMM(b.GetEnd().x) - pcbnew.ToMM(b.GetStart().x),
                pcbnew.ToMM(b.GetEnd().y) - pcbnew.ToMM(b.GetStart().y))
            # Remove shorter one (less routing impact)
            to_remove.add(id(a) if la < lb else id(b))

        removed_tracks = 0
        for a, b in crossings:
            if id(a) in to_remove:
                try:
                    board.Remove(a)
                    removed_tracks += 1
                except Exception:
                    pass
            if id(b) in to_remove:
                try:
                    board.Remove(b)
                    removed_tracks += 1
                except Exception:
                    pass
        print(f"Removed {removed_tracks} crossing tracks")
        removed += removed_tracks

    if removed > 0:
        out = args.pcb if args.in_place else args.pcb.replace(".kicad_pcb", "_cleaned.kicad_pcb")
        board.Save(out)
        print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
