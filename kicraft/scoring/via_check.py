"""Via density and thermal via check."""
import math
import pcbnew
from .base import LayoutCheck, CheckResult, Issue


class ViaCheck(LayoutCheck):
    name = "vias"
    display_name = "Via Analysis"
    weight = 0.10

    def run(self, board, config: dict) -> CheckResult:
        thermal_refs = config.get("thermal_via_refs", [])
        thermal_radius = config.get("thermal_via_radius_mm", 3.0)
        min_thermal = config.get("min_thermal_vias", 4)

        # Board area
        board_rect = board.GetBoardEdgesBoundingBox()
        board_area_cm2 = (pcbnew.ToMM(board_rect.GetWidth()) * pcbnew.ToMM(board_rect.GetHeight())) / 100.0

        # Collect vias
        vias = []
        via_nets = {}
        for track in board.GetTracks():
            if isinstance(track, pcbnew.PCB_VIA):
                pos = track.GetPosition()
                x, y = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
                net = track.GetNetname()
                vias.append({"x": x, "y": y, "net": net, "drill": pcbnew.ToMM(track.GetDrill())})
                via_nets[net] = via_nets.get(net, 0) + 1

        # Thermal via analysis per IC
        thermal_results = {}
        issues = []
        for ref in thermal_refs:
            fp = board.FindFootprintByReference(ref)
            if not fp:
                issues.append(Issue("info", f"Thermal via ref '{ref}' not found on board"))
                thermal_results[ref] = 0
                continue
            fp_pos = fp.GetPosition()
            fx, fy = pcbnew.ToMM(fp_pos.x), pcbnew.ToMM(fp_pos.y)
            nearby = sum(1 for v in vias
                         if math.hypot(v["x"] - fx, v["y"] - fy) <= thermal_radius)
            thermal_results[ref] = nearby
            if nearby < min_thermal:
                issues.append(Issue("warning",
                    f"{ref} has {nearby} vias within {thermal_radius}mm (need >= {min_thermal})",
                    {"ref": ref, "x": round(fx, 2), "y": round(fy, 2)}))

        total_vias = len(vias)
        density = total_vias / board_area_cm2 if board_area_cm2 > 0 else 0

        # Scoring
        # Thermal vias: 40 pts
        if thermal_refs:
            thermal_scores = []
            for ref in thermal_refs:
                count = thermal_results.get(ref, 0)
                if count >= min_thermal:
                    thermal_scores.append(40)
                elif count > 0:
                    thermal_scores.append(40 * count / min_thermal)
                else:
                    thermal_scores.append(0)
            thermal_score = sum(thermal_scores) / len(thermal_scores)
        else:
            thermal_score = 40  # No thermal ICs specified, full marks

        # Density: 30 pts (reasonable range: 2-20 vias/cm2)
        if 2 <= density <= 20:
            density_score = 30
        elif density < 2:
            density_score = 30 * (density / 2)
        else:
            density_score = max(0, 30 - (density - 20) * 1.5)

        # Basic presence: 30 pts
        presence_score = 30 if total_vias > 0 else 0

        score = round(thermal_score + density_score + presence_score, 1)

        return CheckResult(
            score=min(100, score),
            issues=issues,
            metrics={
                "total_vias": total_vias,
                "density_per_cm2": round(density, 2),
                "via_nets": via_nets,
                "thermal_vias": thermal_results,
            },
            summary=f"{total_vias} vias ({density:.1f}/cm2), thermal: {thermal_results}",
        )
