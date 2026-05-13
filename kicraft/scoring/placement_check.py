"""Component placement quality check."""
import pcbnew
from .base import LayoutCheck, CheckResult, Issue


class PlacementCheck(LayoutCheck):
    name = "placement"
    display_name = "Component Placement"
    weight = 0.22

    def run(self, board, config: dict) -> CheckResult:
        util_range = config.get("target_utilization_range", [0.30, 0.70])

        # Board area from edge cuts
        board_rect = board.GetBoardEdgesBoundingBox()
        board_w = pcbnew.ToMM(board_rect.GetWidth())
        board_h = pcbnew.ToMM(board_rect.GetHeight())
        board_area = board_w * board_h

        fps = list(board.Footprints())
        issues = []

        # Collect courtyard bounding boxes (tighter than full bbox which includes silkscreen)
        fp_data = []
        total_fp_area = 0
        out_of_bounds = 0  # footprint-center off-board (mild defect)
        pads_off_board = 0  # ANY pad off the physical board edge (fab-fatal)

        bx1 = pcbnew.ToMM(board_rect.GetX())
        by1 = pcbnew.ToMM(board_rect.GetY())
        bx2 = bx1 + board_w
        by2 = by1 + board_h

        for fp in fps:
            # Try courtyard first, fall back to bounding box
            try:
                courtyard = fp.GetCourtyard(pcbnew.F_CrtYd if fp.GetLayer() == pcbnew.F_Cu else pcbnew.B_CrtYd)
                cbox = courtyard.BBox()
                if cbox.GetWidth() > 0 and cbox.GetHeight() > 0:
                    bbox = cbox
                else:
                    bbox = fp.GetBoundingBox()
            except Exception:
                bbox = fp.GetBoundingBox()
            x1 = pcbnew.ToMM(bbox.GetX())
            y1 = pcbnew.ToMM(bbox.GetY())
            w = pcbnew.ToMM(bbox.GetWidth())
            h = pcbnew.ToMM(bbox.GetHeight())
            ref = fp.GetReferenceAsString()
            fp_data.append({"ref": ref, "x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h})
            total_fp_area += w * h

            # Check if footprint center is inside board outline
            pos = fp.GetPosition()
            px, py = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
            if not (bx1 <= px <= bx2 and by1 <= py <= by2):
                out_of_bounds += 1
                issues.append(Issue("error", f"{ref} center is outside board outline",
                                    {"ref": ref, "x": round(px, 2), "y": round(py, 2)}))

            # Solder pads must stay inside Edge.Cuts. Courtyards on
            # edge-mounted connectors (J*) can legitimately overhang
            # (USB-C tab past the board edge, e.g.), but the COPPER
            # PADS need on-board substrate to solder to. Pad off-board
            # is unrecoverable: fab either cuts the pad in half or
            # leaves it dangling without copper.
            for pad in fp.Pads():
                pbox = pad.GetBoundingBox()
                px1 = pcbnew.ToMM(pbox.GetX())
                py1 = pcbnew.ToMM(pbox.GetY())
                px2 = px1 + pcbnew.ToMM(pbox.GetWidth())
                py2 = py1 + pcbnew.ToMM(pbox.GetHeight())
                overshoot = max(
                    bx1 - px1, by1 - py1, px2 - bx2, py2 - by2, 0.0
                )
                if overshoot > 0.01:
                    pads_off_board += 1
                    pad_name = pad.GetNumber() or "?"
                    issues.append(Issue("error",
                        f"{ref} pad {pad_name} extends {overshoot:.2f}mm past board edge "
                        f"(pads must stay inside Edge.Cuts -- fab-fatal)",
                        {"ref": ref, "pad": pad_name, "overshoot_mm": round(overshoot, 3)}))
                    break  # one issue per footprint is enough; user fixes the whole component

        # Overlap detection (O(n^2) is fine for <100 components)
        # Build set of large mechanical refs to skip expected overlaps
        th_refs = set()
        for fp in fps:
            ref = fp.GetReferenceAsString()
            # Skip overlap checks for through-board mechanical parts:
            # BT = battery holders (batteries live above PCB plane, not in 2D footprint),
            # H = mounting holes, J = connectors (edge-mounted, courtyard extends off-board)
            if ref.startswith(("BT", "H", "J")):
                th_refs.add(ref)

        overlaps = 0
        for i in range(len(fp_data)):
            for j in range(i + 1, len(fp_data)):
                a, b = fp_data[i], fp_data[j]
                # Skip overlaps between through-hole/mechanical parts
                if a["ref"] in th_refs and b["ref"] in th_refs:
                    continue
                if a["x1"] < b["x2"] and a["x2"] > b["x1"] and a["y1"] < b["y2"] and a["y2"] > b["y1"]:
                    ox = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
                    oy = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
                    if ox * oy > 2.0:
                        overlaps += 1
                        issues.append(Issue("error",
                            f"{a['ref']} and {b['ref']} overlap ({ox:.1f}x{oy:.1f}mm)",
                            {"ref": f"{a['ref']},{b['ref']}"}))

        utilization = total_fp_area / board_area if board_area > 0 else 0

        # Scoring. Any pad off-board is fab-fatal -- the layout cannot
        # be manufactured. We hard-zero the placement score so the
        # weighted total collapses (placement weight = 0.22, so any
        # off-board pad cuts at minimum 22 points off the overall
        # leaf score regardless of other checks).
        if pads_off_board > 0:
            return CheckResult(
                score=0.0,
                issues=issues,
                metrics={
                    "board_area_mm2": round(board_area, 1),
                    "footprint_area_mm2": round(total_fp_area, 1),
                    "utilization": round(utilization, 3),
                    "overlaps": overlaps,
                    "out_of_bounds": out_of_bounds,
                    "pads_off_board": pads_off_board,
                    "footprint_count": len(fps),
                },
                summary=f"FAB-FATAL: {pads_off_board} component(s) with pads past Edge.Cuts",
            )

        overlap_score = 40.0 * (1.0 / (1.0 + overlaps))
        bounds_score = 20.0 * (1.0 / (1.0 + out_of_bounds))

        if util_range[0] <= utilization <= util_range[1]:
            util_score = 40
        elif utilization < util_range[0]:
            util_score = max(0, 40 * (utilization / util_range[0]))
        else:
            util_score = max(0, 40 * (1 - (utilization - util_range[1]) / (1.0 - util_range[1])))

        score = overlap_score + bounds_score + util_score

        return CheckResult(
            score=round(score, 1),
            issues=issues,
            metrics={
                "board_area_mm2": round(board_area, 1),
                "footprint_area_mm2": round(total_fp_area, 1),
                "utilization": round(utilization, 3),
                "overlaps": overlaps,
                "out_of_bounds": out_of_bounds,
                "pads_off_board": pads_off_board,
                "footprint_count": len(fps),
            },
            summary=f"{len(fps)} components, {utilization:.0%} utilization, {overlaps} overlaps",
        )
