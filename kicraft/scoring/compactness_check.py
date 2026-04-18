"""Board compactness check — penalizes wasted space."""
import pcbnew
from .base import LayoutCheck, CheckResult, Issue


class CompactnessCheck(LayoutCheck):
    name = "compactness"
    display_name = "Board Compactness"
    weight = 0.0  # advisory for now, user adjusts when ready

    def run(self, board, config: dict) -> CheckResult:
        # Board dimensions
        rect = board.GetBoardEdgesBoundingBox()
        board_w = pcbnew.ToMM(rect.GetWidth())
        board_h = pcbnew.ToMM(rect.GetHeight())
        board_area = board_w * board_h

        # Compute tight bounding box of all component centers
        fps = list(board.Footprints())
        if not fps:
            return CheckResult(score=0, issues=[], metrics={}, summary="No footprints")

        xs = []
        ys = []
        for fp in fps:
            pos = fp.GetPosition()
            xs.append(pcbnew.ToMM(pos.x))
            ys.append(pcbnew.ToMM(pos.y))

        # Component bounding box with margin
        margin = 3.0  # mm margin around outermost components
        comp_x1 = min(xs) - margin
        comp_y1 = min(ys) - margin
        comp_x2 = max(xs) + margin
        comp_y2 = max(ys) + margin
        comp_w = comp_x2 - comp_x1
        comp_h = comp_y2 - comp_y1
        comp_bbox_area = comp_w * comp_h

        # Ratio: how much of the board is actually used by components
        fill_ratio = comp_bbox_area / board_area if board_area > 0 else 0

        # Dead zone analysis: split board into grid, find empty cells
        grid_size = 5.0  # mm
        cols = max(1, int(board_w / grid_size))
        rows = max(1, int(board_h / grid_size))
        grid = [[False] * cols for _ in range(rows)]

        bx = pcbnew.ToMM(rect.GetX())
        by = pcbnew.ToMM(rect.GetY())

        for fp in fps:
            pos = fp.GetPosition()
            px = pcbnew.ToMM(pos.x) - bx
            py = pcbnew.ToMM(pos.y) - by
            bbox = fp.GetBoundingBox()
            fw = pcbnew.ToMM(bbox.GetWidth())
            fh = pcbnew.ToMM(bbox.GetHeight())

            # Mark grid cells covered by this footprint
            c1 = max(0, int((px - fw/2) / grid_size))
            c2 = min(cols - 1, int((px + fw/2) / grid_size))
            r1 = max(0, int((py - fh/2) / grid_size))
            r2 = min(rows - 1, int((py + fh/2) / grid_size))
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    grid[r][c] = True

        total_cells = rows * cols
        occupied_cells = sum(sum(row) for row in grid)
        total_cells - occupied_cells
        cell_fill = occupied_cells / total_cells if total_cells > 0 else 0

        issues = []

        # Check for large empty regions (>20% of board)
        if cell_fill < 0.3:
            issues.append(Issue("warning",
                f"Only {cell_fill:.0%} of board grid cells have components — "
                f"board may be oversized"))

        # Check if board could be smaller
        if fill_ratio < 0.5:
            potential_w = comp_w
            potential_h = comp_h
            issues.append(Issue("info",
                f"Components fit in {potential_w:.0f}x{potential_h:.0f}mm "
                f"({fill_ratio:.0%} of {board_w:.0f}x{board_h:.0f}mm board)"))

        # Scoring
        # fill_ratio: what fraction of board area is the component bbox
        # cell_fill: what fraction of grid cells have components
        if fill_ratio >= 0.8:
            bbox_score = 50
        elif fill_ratio >= 0.5:
            bbox_score = 50 * (fill_ratio - 0.3) / 0.5
        else:
            bbox_score = max(0, 50 * fill_ratio / 0.5)

        if cell_fill >= 0.5:
            grid_score = 50
        elif cell_fill >= 0.2:
            grid_score = 50 * (cell_fill - 0.1) / 0.4
        else:
            grid_score = max(0, 50 * cell_fill / 0.2)

        score = round(bbox_score + grid_score, 1)

        return CheckResult(
            score=min(100, score),
            issues=issues,
            metrics={
                "board_mm": f"{board_w:.1f}x{board_h:.1f}",
                "board_area_mm2": round(board_area, 1),
                "component_bbox_mm": f"{comp_w:.1f}x{comp_h:.1f}",
                "component_bbox_area_mm2": round(comp_bbox_area, 1),
                "fill_ratio": round(fill_ratio, 3),
                "grid_fill": round(cell_fill, 3),
                "grid_cells": f"{occupied_cells}/{total_cells}",
            },
            summary=f"Board {board_w:.0f}x{board_h:.0f}mm, "
                    f"components fill {fill_ratio:.0%}, "
                    f"grid {cell_fill:.0%} occupied",
        )
