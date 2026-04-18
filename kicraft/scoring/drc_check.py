"""DRC check using kicad-cli for accurate results."""
import os
import re
import subprocess
import tempfile

import pcbnew
from .base import LayoutCheck, CheckResult, Issue


# Violations that are critical for board function
CRITICAL_TYPES = {"shorting_items", "tracks_crossing", "unconnected_items"}
# Violations that affect manufacturability
MAJOR_TYPES = {"clearance", "hole_clearance", "hole_to_hole", "copper_edge_clearance"}
# Cosmetic / informational
MINOR_TYPES = {"solder_mask_bridge", "silk_overlap", "silk_over_copper",
               "silk_edge_clearance", "courtyards_overlap", "via_dangling",
               "track_dangling", "lib_footprint_mismatch", "lib_footprint_issues"}


class DRCCheck(LayoutCheck):
    name = "drc_markers"
    display_name = "DRC Violations"
    weight = 0.35

    def run(self, board, config: dict) -> CheckResult:
        pcb_path = config.get("_pcb_path", "")
        if not pcb_path:
            return self._fallback_markers(board)

        return self._run_kicad_cli(pcb_path)

    def _run_kicad_cli(self, pcb_path):
        """Run full DRC via kicad-cli and parse the text report."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            report_path = f.name

        try:
            subprocess.run(
                ["kicad-cli", "pcb", "drc", "-o", report_path, pcb_path],
                capture_output=True, text=True, timeout=60,
            )
            with open(report_path) as f:
                report = f.read()
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return CheckResult(
                score=0, issues=[Issue("error", f"kicad-cli DRC failed: {e}")],
                metrics={}, summary="DRC failed",
            )
        finally:
            try:
                os.remove(report_path)
            except OSError:
                pass

        return self._parse_report(report)

    def _parse_report(self, report):
        """Parse kicad-cli DRC text report."""
        counts = {}  # type -> count
        issues = []

        # Match lines like: [shorting_items]: Items shorting two nets (...)
        for line in report.splitlines():
            m = re.match(r'^\[(\w+)\]:\s+(.+)', line)
            if not m:
                continue
            vtype = m.group(1)
            msg = m.group(2)
            counts[vtype] = counts.get(vtype, 0) + 1

            if vtype in CRITICAL_TYPES:
                issues.append(Issue("error", f"[{vtype}] {msg}"))
            elif vtype in MAJOR_TYPES:
                issues.append(Issue("warning", f"[{vtype}] {msg}"))
            # Skip minor issues from issue list to save tokens

        # Also count unconnected items section
        uc_match = re.search(r'\*\* Found (\d+) unconnected', report)
        if uc_match:
            uc_count = int(uc_match.group(1))
            if "unconnected_items" not in counts:
                counts["unconnected_items"] = uc_count

        # Scoring — heavily penalize critical issues
        critical = sum(counts.get(t, 0) for t in CRITICAL_TYPES)
        major = sum(counts.get(t, 0) for t in MAJOR_TYPES)
        minor = sum(counts.get(t, 0) for t in MINOR_TYPES)
        total = sum(counts.values())

        # Shorts and unconnected are catastrophic
        shorts = counts.get("shorting_items", 0)
        unconnected = counts.get("unconnected_items", 0)
        crossings = counts.get("tracks_crossing", 0)

        # Logarithmic scoring so improvements always register
        # 0 issues = 100, 1 = ~80, 5 = ~50, 20 = ~20, 50+ = ~5
        import math

        def issue_score(count, weight):
            """Score component: 0 issues=weight, scales down logarithmically."""
            if count == 0:
                return weight
            return max(0, weight * (1 - math.log10(1 + count) / math.log10(100)))

        score = 0.0
        score += issue_score(shorts, 40)         # 40 pts for no shorts
        score += issue_score(unconnected, 30)    # 30 pts for all connected
        score += issue_score(crossings, 15)      # 15 pts for no crossings
        score += issue_score(major, 10)          # 10 pts for clearance
        score += issue_score(minor, 5)           # 5 pts for cosmetic
        score = round(score, 1)

        # Truncate issues list to save tokens (keep worst 20)
        issues = [i for i in issues if i.severity == "error"][:20]

        return CheckResult(
            score=score,
            issues=issues,
            metrics={
                "total_violations": total,
                "by_type": counts,
                "critical": critical,
                "major": major,
                "minor": minor,
                "shorts": shorts,
                "unconnected": unconnected,
                "crossings": crossings,
            },
            summary=(f"{shorts} shorts, {unconnected} unconnected, "
                     f"{crossings} crossings, {major} clearance, {minor} cosmetic"),
        )

    def _fallback_markers(self, board):
        """Fallback: read stored DRC markers from board file."""
        markers = list(board.Markers())
        issues = []
        errors = 0
        warnings = 0

        for marker in markers:
            pos = marker.GetPosition()
            x, y = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
            rc = marker.GetRCItem()
            msg = rc.GetErrorMessage() if rc else "Unknown"
            sev = marker.GetSeverity()

            if sev == 0:
                errors += 1
                issues.append(Issue("error", msg, {"x": round(x, 2), "y": round(y, 2)}))
            elif sev == 1:
                warnings += 1
                issues.append(Issue("warning", msg, {"x": round(x, 2), "y": round(y, 2)}))

        score = max(0, 100 - errors * 20 - warnings * 5)

        if not markers:
            issues.append(Issue("info", "No stored DRC markers — run kicad-cli for accurate results"))

        return CheckResult(
            score=score, issues=issues,
            metrics={"errors": errors, "warnings": warnings, "total_markers": len(markers)},
            summary=f"{errors} errors, {warnings} warnings" if markers else "No stored markers",
        )
