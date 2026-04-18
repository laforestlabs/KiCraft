"""Visual analysis check — renders PCB to images for human/AI review.

This check renders the PCB to PNG images using kicad-cli and stores them
alongside the scoring results. It does NOT auto-score from pixels — the
images are meant to be reviewed by Claude (multimodal) or a human to catch
issues that programmatic checks miss:

- Trace routing aesthetics (90° corners, unnecessary detours)
- Ground plane fragmentation visible on B.Cu render
- Component grouping / logical flow
- Silkscreen readability and overlap
- Thermal pad exposure and via placement
- General "does this look right" sanity check

The check always returns score=None (excluded from weighted average) and
attaches image paths to the result for downstream consumption.
"""
import os
import subprocess
import sys

from .base import LayoutCheck, CheckResult, Issue

# Minimal views for scoring context — keep render time low
SCORE_VIEWS = {
    "front_all": {
        "layers": "F.Cu,F.SilkS,F.Mask,Edge.Cuts",
        "mirror": False,
    },
    "back_copper": {
        "layers": "B.Cu,Edge.Cuts",
        "mirror": True,
    },
    "copper_both": {
        "layers": "F.Cu,B.Cu,Edge.Cuts",
        "mirror": False,
    },
}


class VisualCheck(LayoutCheck):
    name = "visual"
    display_name = "Visual Analysis"
    weight = 0.0  # not scored — advisory only

    def run(self, board, config: dict) -> CheckResult:
        pcb_path = config.get("_pcb_path", "")
        output_dir = config.get("_render_dir", "")

        if not pcb_path or not output_dir:
            return CheckResult(
                score=0,
                issues=[Issue("info", "Visual check skipped — no PCB path or render dir in config")],
                metrics={},
                summary="Skipped",
            )

        os.makedirs(output_dir, exist_ok=True)
        rendered = {}
        issues = []

        for view_name, view_cfg in SCORE_VIEWS.items():
            png_path = self._render_one(pcb_path, view_name, view_cfg, output_dir)
            if png_path:
                rendered[view_name] = png_path
            else:
                issues.append(Issue("warning", f"Failed to render {view_name}"))

        if not rendered:
            issues.append(Issue("error", "No views rendered — check kicad-cli and ImageMagick"))

        return CheckResult(
            score=0,  # advisory — not factored into overall
            issues=issues,
            metrics={
                "rendered_views": list(rendered.keys()),
                "render_paths": rendered,
                "view_count": len(rendered),
            },
            summary=f"{len(rendered)} views rendered for visual review",
        )

    @staticmethod
    def _render_one(pcb_path, name, cfg, output_dir, dpi=300, max_px=2000):
        svg_path = os.path.join(output_dir, f"{name}.svg")
        png_path = os.path.join(output_dir, f"{name}.png")

        cmd = [
            "kicad-cli", "pcb", "export", "svg",
            "--layers", cfg["layers"],
            "--mode-single", "--fit-page-to-board",
            "--exclude-drawing-sheet", "--drill-shape-opt", "2",
            "-o", svg_path,
        ]
        if cfg.get("mirror"):
            cmd.append("--mirror")
        cmd.append(pcb_path)

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return None

        r = subprocess.run([
            "magick", "-density", str(dpi),
            svg_path, "-resize", f"{max_px}x{max_px}",
            png_path,
        ], capture_output=True, text=True)

        # Clean up SVG
        try:
            os.remove(svg_path)
        except OSError:
            pass

        return png_path if r.returncode == 0 else None
