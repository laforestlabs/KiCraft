"""Net connectivity check."""
from .base import LayoutCheck, CheckResult, Issue


class ConnectivityCheck(LayoutCheck):
    name = "connectivity"
    display_name = "Net Connectivity"
    weight = 0.17

    def run(self, board, config: dict) -> CheckResult:
        board.BuildConnectivity()

        net_pads = {}
        unassigned = 0

        for pad in board.GetPads():
            net = pad.GetNetname()
            if net == "":
                # Ignore mounting hole pads and pads on footprints with
                # "unconnected-" nets (intentionally unused pins like USB data)
                fp = pad.GetParentFootprint()
                ref = fp.GetReferenceAsString() if fp else "?"
                # Check if this footprint has any unconnected-* nets (deliberate no-connects)
                has_deliberate_nc = False
                if fp:
                    for other_pad in fp.Pads():
                        if other_pad.GetNetname().startswith("unconnected-"):
                            has_deliberate_nc = True
                            break
                # Also ignore battery holder mounting pads (empty pad number = mechanical)
                pad_num = pad.GetNumber()
                if not has_deliberate_nc and not ref.startswith("H") and pad_num != "":
                    unassigned += 1
                continue
            if net.startswith("unconnected-"):
                continue  # intentional no-connects, don't count as real nets
            net_pads[net] = net_pads.get(net, 0) + 1

        single_pad_nets = []
        for name, count in sorted(net_pads.items()):
            if count == 1 and not name.startswith("unconnected-"):
                single_pad_nets.append(name)

        issues = []
        for net in single_pad_nets:
            issues.append(Issue("warning", f"Net '{net}' has only 1 pad (likely unconnected)", {"net": net}))

        if unassigned > 0:
            issues.append(Issue("error", f"{unassigned} pad(s) have no net assignment"))

        score = max(0, 100 - len(single_pad_nets) * 10 - unassigned * 5)

        return CheckResult(
            score=score,
            issues=issues,
            metrics={
                "total_nets": len(net_pads),
                "single_pad_nets": len(single_pad_nets),
                "unassigned_pads": unassigned,
            },
            summary=f"{len(net_pads)} nets, {len(single_pad_nets)} single-pad, {unassigned} unassigned pads",
        )
