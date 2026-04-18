"""Functional footprint orientation check.

Scores whether connectors, switches, test points, and other access-critical
components are oriented correctly for physical use — e.g., USB connectors
should face outward from a board edge, not inward.
"""
import pcbnew
from .base import LayoutCheck, CheckResult, Issue


ORIENTATION_RULES = {
    "J": {
        "description": "Connector should face outward from nearest board edge",
        "max_edge_distance_mm": 5.0,
        "check_facing": True,
    },
    "SW": {
        "description": "Switch should be accessible near board edge",
        "max_edge_distance_mm": 10.0,
        "check_facing": False,
    },
    "TP": {
        "description": "Test point should be accessible",
        "max_edge_distance_mm": 15.0,
        "check_facing": False,
    },
}

# Expected outward direction per edge (board-space angle)
_OUTWARD = {"left": 180, "right": 0, "top": 270, "bottom": 90}


def detect_opening_direction_board(fp) -> float | None:
    """Detect which direction the connector opening faces in BOARD space.

    All coordinates come directly from pcbnew in board space — no rotation
    math is needed.  Compares pad bounding box to body bounding box
    (courtyard + fab layer graphics).  The side where the body extends
    furthest beyond the pads is the opening / mating face.

    Returns 0 (+X/right), 90 (+Y/down), 180 (-X/left), 270 (-Y/up),
    or None if not clearly directional.
    """
    pad_xs = [pcbnew.ToMM(p.GetPosition().x) for p in fp.Pads()]
    pad_ys = [pcbnew.ToMM(p.GetPosition().y) for p in fp.Pads()]
    if not pad_xs:
        return None

    body_xs, body_ys = [], []
    cy_layer = pcbnew.F_CrtYd if fp.GetLayer() == pcbnew.F_Cu else pcbnew.B_CrtYd
    fab_layer = pcbnew.F_Fab if fp.GetLayer() == pcbnew.F_Cu else pcbnew.B_Fab

    for item in fp.GraphicalItems():
        if item.GetLayer() not in (cy_layer, fab_layer):
            continue
        try:
            body_xs.append(pcbnew.ToMM(item.GetStart().x))
            body_xs.append(pcbnew.ToMM(item.GetEnd().x))
            body_ys.append(pcbnew.ToMM(item.GetStart().y))
            body_ys.append(pcbnew.ToMM(item.GetEnd().y))
        except Exception:
            continue

    if not body_xs:
        return None

    extensions = {
        0:   max(body_xs) - max(pad_xs),   # +X (right)
        180: min(pad_xs)  - min(body_xs),   # -X (left)
        90:  max(body_ys) - max(pad_ys),    # +Y (down)
        270: min(pad_ys)  - min(body_ys),   # -Y (up)
    }

    ranked = sorted(extensions.items(), key=lambda kv: kv[1], reverse=True)
    best_dir, best_ext = ranked[0]
    _, second_ext = ranked[1]

    if best_ext >= 1.0 and (best_ext - second_ext) >= 0.5:
        return best_dir

    # Fallback: "PCB Edge" / "Board Edge" text on Dwgs.User
    pad_cx = (min(pad_xs) + max(pad_xs)) / 2
    pad_cy = (min(pad_ys) + max(pad_ys)) / 2
    for item in fp.GraphicalItems():
        if item.GetLayer() != pcbnew.Dwgs_User:
            continue
        try:
            text = item.GetText()
        except Exception:
            continue
        if not text or "edge" not in text.lower():
            continue
        tp = item.GetPosition()
        off_x = pcbnew.ToMM(tp.x) - pad_cx
        off_y = pcbnew.ToMM(tp.y) - pad_cy
        if abs(off_x) > abs(off_y):
            return 0 if off_x > 0 else 180
        else:
            return 90 if off_y > 0 else 270

    return None


class OrientationCheck(LayoutCheck):
    name = "orientation"
    display_name = "Footprint Orientation"
    weight = 0.0  # advisory for now

    def run(self, board, config: dict) -> CheckResult:
        rect = board.GetBoardEdgesBoundingBox()
        bx1 = pcbnew.ToMM(rect.GetX())
        by1 = pcbnew.ToMM(rect.GetY())
        bx2 = bx1 + pcbnew.ToMM(rect.GetWidth())
        by2 = by1 + pcbnew.ToMM(rect.GetHeight())

        issues = []
        checked = 0
        passed = 0

        for fp in board.Footprints():
            ref = fp.GetReferenceAsString()

            rule = None
            for prefix, r in ORIENTATION_RULES.items():
                if ref.startswith(prefix):
                    rule = r
                    break
            if not rule:
                continue

            checked += 1
            pos = fp.GetPosition()
            px = pcbnew.ToMM(pos.x)
            py = pcbnew.ToMM(pos.y)
            rot = fp.GetOrientationDegrees() % 360

            edges = {
                "left": px - bx1,
                "right": bx2 - px,
                "top": py - by1,
                "bottom": by2 - py,
            }
            nearest_edge = min(edges, key=edges.get)
            edge_dist = edges[nearest_edge]

            max_dist = rule["max_edge_distance_mm"]
            edge_ok = edge_dist <= max_dist

            facing_ok = True
            if rule.get("check_facing") and edge_ok:
                opening_board = detect_opening_direction_board(fp)
                if opening_board is not None:
                    expected = _OUTWARD[nearest_edge]
                    angle_err = abs(opening_board - expected) % 360
                    if angle_err > 180:
                        angle_err = 360 - angle_err
                    facing_ok = angle_err <= 45

            if edge_ok and facing_ok:
                passed += 1
            else:
                severity = "error" if ref.startswith("J") else "warning"
                if not edge_ok:
                    issues.append(Issue(severity,
                        f"{ref} is {edge_dist:.1f}mm from nearest edge ({nearest_edge}), "
                        f"max {max_dist}mm — {rule['description']}",
                        {"ref": ref, "x": round(px, 1), "y": round(py, 1)}))
                elif not facing_ok:
                    issues.append(Issue(severity,
                        f"{ref} at {rot:.0f}° near {nearest_edge} edge — "
                        f"opening faces wrong direction. {rule['description']}",
                        {"ref": ref, "rotation": rot, "nearest_edge": nearest_edge}))

        if checked == 0:
            return CheckResult(score=100, issues=[], metrics={},
                               summary="No orientation-critical components found")

        score = round(100 * passed / checked, 1)

        return CheckResult(
            score=score,
            issues=issues,
            metrics={
                "checked": checked,
                "passed": passed,
                "failed": checked - passed,
            },
            summary=f"{passed}/{checked} access-critical components correctly oriented",
        )
