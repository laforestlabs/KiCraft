"""Routing geometry and efficiency check."""
import math
import pcbnew
from .base import LayoutCheck, CheckResult, Issue


class GeometryCheck(LayoutCheck):
    name = "geometry"
    display_name = "Routing Efficiency"
    weight = 0.10

    def run(self, board, config: dict) -> CheckResult:
        power_nets = config.get("power_nets", [])

        # Collect trace lengths per net
        net_traces = {}  # net -> {"total_length_mm": float, "segments": int}
        for track in board.GetTracks():
            if isinstance(track, pcbnew.PCB_VIA):
                continue
            net = track.GetNetname()
            start = track.GetStart()
            end = track.GetEnd()
            length = math.hypot(
                pcbnew.ToMM(end.x) - pcbnew.ToMM(start.x),
                pcbnew.ToMM(end.y) - pcbnew.ToMM(start.y),
            )
            if net not in net_traces:
                net_traces[net] = {"total_length_mm": 0, "segments": 0}
            net_traces[net]["total_length_mm"] += length
            net_traces[net]["segments"] += 1

        # Collect pad positions per net for optimal distance estimate
        net_pads = {}
        for pad in board.GetPads():
            net = pad.GetNetname()
            if not net:
                continue
            pos = pad.GetPosition()
            x, y = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
            if net not in net_pads:
                net_pads[net] = []
            net_pads[net].append((x, y))

        # Compute routing efficiency per net
        issues = []
        ratios = []
        net_metrics = {}

        for net, data in net_traces.items():
            pads = net_pads.get(net, [])
            if len(pads) < 2:
                continue

            # Minimum spanning distance estimate: sum of distances in MST
            # Approximate with total pairwise min distance (greedy nearest-neighbor)
            optimal = self._greedy_mst_length(pads)
            if optimal < 0.1:
                continue

            actual = data["total_length_mm"]
            ratio = actual / optimal
            ratios.append(ratio)
            net_metrics[net] = {
                "actual_mm": round(actual, 2),
                "optimal_mm": round(optimal, 2),
                "ratio": round(ratio, 2),
                "segments": data["segments"],
            }

            if ratio > 3.0:
                is_power = any(p in net for p in power_nets)
                sev = "warning" if is_power else "info"
                issues.append(Issue(sev,
                    f"Net '{net}' routing {ratio:.1f}x optimal ({actual:.1f}mm vs {optimal:.1f}mm est.)",
                    {"net": net}))

        avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0

        # Scoring
        # Efficiency: 40 pts
        if avg_ratio <= 1.5:
            efficiency_score = 40
        elif avg_ratio <= 3.0:
            efficiency_score = 40 * (1 - (avg_ratio - 1.5) / 1.5)
        else:
            efficiency_score = 0

        # No orphaned segments (nets with traces but no pads): 30 pts
        orphaned = sum(1 for net in net_traces if net not in net_pads or len(net_pads.get(net, [])) == 0)
        orphan_score = max(0, 30 - orphaned * 10)

        # Total trace count reasonable: 30 pts
        total_segments = sum(d["segments"] for d in net_traces.values())
        segment_score = 30 if total_segments > 0 else 0

        score = round(efficiency_score + orphan_score + segment_score, 1)

        return CheckResult(
            score=min(100, score),
            issues=issues,
            metrics={
                "total_trace_length_mm": round(sum(d["total_length_mm"] for d in net_traces.values()), 1),
                "total_segments": total_segments,
                "avg_efficiency_ratio": round(avg_ratio, 2),
                "nets_analyzed": len(net_metrics),
                "nets": net_metrics,
            },
            summary=f"{total_segments} segments, avg efficiency ratio {avg_ratio:.2f}x",
        )

    @staticmethod
    def _greedy_mst_length(pads):
        """Greedy nearest-neighbor MST approximation."""
        if len(pads) < 2:
            return 0
        remaining = list(pads)
        current = remaining.pop(0)
        total = 0
        while remaining:
            dists = [(math.hypot(p[0] - current[0], p[1] - current[1]), i)
                     for i, p in enumerate(remaining)]
            d, idx = min(dists)
            total += d
            current = remaining.pop(idx)
        return total
