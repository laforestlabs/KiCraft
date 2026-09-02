"""Visual analysis check — renders PCB to images for human/AI review.

This check renders the PCB to PNG images via the unified
``kicraft.render.render_views`` pipeline and stores them alongside the
scoring results. It does NOT auto-score from pixels — the images are
meant to be reviewed by a multimodal agent or a human to catch issues that
programmatic checks miss:

- Trace routing aesthetics (90° corners, unnecessary detours)
- Ground plane fragmentation visible on B.Cu render
- Component grouping / logical flow
- Silkscreen readability and overlap
- Thermal pad exposure and via placement
- General "does this look right" sanity check

The check always returns score=None (excluded from weighted average)
and attaches image paths to the result for downstream consumption.

Layer presets and post-processing come from ``kicraft.render.VIEWS``
so the score-time previews are bit-for-bit identical to the monitor
and CLI previews — the visual check cannot drift from what the user
sees in the GUI.
"""
import os
from pathlib import Path

from kicraft.render import render_views

from .base import LayoutCheck, CheckResult, Issue

# Subset of VIEWS rendered at score time. Names are the keys in
# ``kicraft.render.VIEWS``; presets (layers, mirror, post) come from
# there so any tuning happens in one place.
SCORE_VIEW_NAMES = ["front_all", "back_copper", "copper_both"]


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
        results = render_views(
            Path(pcb_path),
            Path(output_dir),
            views=SCORE_VIEW_NAMES,
        )
        rendered = {name: str(path) for name, path in results.items()}
        issues = [
            Issue("warning", f"Failed to render {name}")
            for name in SCORE_VIEW_NAMES
            if name not in rendered
        ]
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
