"""Trace width compliance check."""
import pcbnew
from .base import LayoutCheck, CheckResult, Issue


class TraceWidthCheck(LayoutCheck):
    name = "trace_widths"
    display_name = "Trace Width Compliance"
    weight = 0.11

    def run(self, board, config: dict) -> CheckResult:
        power_nets = config.get("power_nets", [])
        power_min = config.get("power_trace_min_mm", 0.127)
        signal_min = config.get("signal_trace_min_mm", 0.127)

        nets = {}  # net_name -> {"widths": [], "is_power": bool}
        for track in board.GetTracks():
            if isinstance(track, pcbnew.PCB_VIA):
                continue
            net = track.GetNetname()
            w_mm = pcbnew.ToMM(track.GetWidth())
            if net not in nets:
                is_power = any(p in net for p in power_nets)
                nets[net] = {"widths": [], "is_power": is_power}
            nets[net]["widths"].append(w_mm)

        issues = []
        power_violations = 0
        signal_violations = 0

        for net_name, data in sorted(nets.items()):
            min_w = min(data["widths"])
            if data["is_power"] and min_w < power_min:
                power_violations += 1
                issues.append(Issue(
                    severity="error",
                    message=f"Power net '{net_name}' has trace {min_w:.3f}mm < {power_min}mm minimum",
                    location={"net": net_name},
                ))
            elif not data["is_power"] and min_w < signal_min:
                signal_violations += 1
                issues.append(Issue(
                    severity="warning",
                    message=f"Signal net '{net_name}' has trace {min_w:.3f}mm < {signal_min}mm minimum",
                    location={"net": net_name},
                ))

        score = max(0, 100 - power_violations * 15 - signal_violations * 5)

        net_metrics = {}
        for name, data in nets.items():
            ws = data["widths"]
            net_metrics[name] = {
                "min_mm": round(min(ws), 3),
                "max_mm": round(max(ws), 3),
                "mean_mm": round(sum(ws) / len(ws), 3),
                "count": len(ws),
                "is_power": data["is_power"],
            }

        return CheckResult(
            score=score,
            issues=issues,
            metrics={
                "total_traces": sum(len(d["widths"]) for d in nets.values()),
                "power_violations": power_violations,
                "signal_violations": signal_violations,
                "nets": net_metrics,
            },
            summary=f"{power_violations} power violations, {signal_violations} signal violations across {len(nets)} nets",
        )
