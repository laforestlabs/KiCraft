"""Regression guard: the GUI's PCB render must land every footprint
where the .kicad_pcb says it lives.

The GUI's parent / leaf preview PNG is produced by
``kicraft.render.render_views`` (see ``kicraft/render/pcb_renderer.py``).
That pipeline calls ``kicad-cli pcb export svg`` with a custom layer
subset and then rewrites the SVG viewBox to the Edge.Cuts AABB before
rasterizing. Two failure modes are easy to introduce there:

1. A change to the viewBox rewrite shifts every component by the
   board's origin offset (visible as "all components moved by ~100 mm
   to the left/up").

2. A change to the layer set or the silkscreen stamping makes the
   render visually agree with the file at one zoom level but diverge
   at another (e.g. clipping silkscreen text that hangs off Edge.Cuts).

This test renders a real saved board through the same code path the GUI
uses, then verifies that for each footprint in the .kicad_pcb the
projected pixel position lands on a non-background pixel in the
rendered PNG. The check is intentionally loose -- we only assert that
*something* (silk, copper, or pad) shows up where the file says the
component is centered. It catches "all the components got shifted by
a constant" without overfitting to layer colors or DPI.

Skipped when ``kicad-cli`` / ``magick`` / ``pcbnew`` are missing so the
suite still runs on developer machines without KiCad installed.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest


pcbnew = pytest.importorskip("pcbnew")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from kicraft.render import render_views  # noqa: E402
from kicraft.render.edge_cuts import parse_edge_cuts_aabb  # noqa: E402


REPO = Path(__file__).resolve().parents[1]

# A real composed parent board from the manual-runs fixtures. Locked to
# a specific subcircuit hash so the test is reproducible -- the
# autoexperiment writes parent_placed.kicad_pcb here on every
# composed round and the file's positions are stable for a given seed.
PARENT_PCB = (
    REPO
    / "tests/manual-runs/esp32motionsensor/generated/ESP32_MOTION_SENSOR"
    / ".experiments/subcircuits/subcircuit__8a5edab282"
    / "parent_placed.kicad_pcb"
)


def _have_external_tools() -> bool:
    return shutil.which("kicad-cli") is not None and shutil.which("magick") is not None


def _extract_footprint_centers(pcb: Path) -> list[tuple[str, float, float]]:
    """Return ``[(ref, x_mm, y_mm), ...]`` for front-side footprints in
    the .kicad_pcb. Uses ``pcbnew.LoadBoard`` so the test's ground truth
    is the same parser KiCad itself uses -- no inline s-expression walker
    to drift against format changes, no regex fragility around parens
    inside property strings or long property tables. Back-side footprints
    are excluded because the front_all render preset doesn't draw
    B.SilkS / B.CrtYd, so projecting their centers would compare against
    pure substrate."""
    board = pcbnew.LoadBoard(str(pcb))
    out: list[tuple[str, float, float]] = []
    for fp in board.GetFootprints():
        if fp.IsFlipped():
            continue
        pos = fp.GetPosition()
        out.append((
            fp.GetReference(),
            pcbnew.ToMM(pos.x),
            pcbnew.ToMM(pos.y),
        ))
    return out


@pytest.mark.skipif(
    not _have_external_tools(), reason="kicad-cli / magick not on PATH"
)
@pytest.mark.skipif(
    not PARENT_PCB.is_file(),
    reason=f"fixture PCB missing: {PARENT_PCB.relative_to(REPO)}",
)
def test_gui_render_projects_footprints_to_correct_pixels(tmp_path):
    """For every footprint inside Edge.Cuts, the pixel at its projected
    position in the GUI render must be non-background -- i.e. some
    geometry (copper, silk, pad) was actually drawn where the file
    centers the part. Catches the "all components shifted by a
    constant" class of viewBox / coordinate-rewrite bugs."""
    aabb = parse_edge_cuts_aabb(PARENT_PCB)
    assert aabb is not None, "fixture PCB has no Edge.Cuts geometry"
    x0, y0, x1, y1 = aabb
    w_mm, h_mm = x1 - x0, y1 - y0
    assert w_mm > 0 and h_mm > 0, (
        f"degenerate Edge.Cuts AABB {aabb} -- fixture is broken"
    )

    results = render_views(PARENT_PCB, tmp_path, views=["front_all"])
    gui_png = results.get("front_all")
    assert gui_png is not None and Path(gui_png).is_file(), \
        "render_views did not produce front_all.png"

    img = np.asarray(Image.open(gui_png).convert("RGB"))
    ih, iw = img.shape[:2]

    # Background = MonitorStyle pink substrate, rgba(255,182,193,*) with
    # alpha 0.5 composited over transparent. After conversion to RGB the
    # substrate reads as approximately (255, 218, 224). Anything that
    # isn't substrate (copper red, silk yellow, pad black, label cyan)
    # diverges from that color. Use a generous tolerance -- the test is
    # only asking "is this pixel painted with NON-substrate content?".
    substrate_rgb = np.array([255, 218, 224], dtype=np.int16)

    failures: list[str] = []
    checked = 0
    for ref, x_mm, y_mm in _extract_footprint_centers(PARENT_PCB):
        if not (x0 <= x_mm <= x1 and y0 <= y_mm <= y1):
            continue
        px = int(round((x_mm - x0) / w_mm * iw))
        py = int(round((y_mm - y0) / h_mm * ih))
        # Sample a small neighborhood so a 1-pixel pad doesn't get missed
        # by a single-pixel probe.
        radius = 6
        x_lo, x_hi = max(0, px - radius), min(iw, px + radius + 1)
        y_lo, y_hi = max(0, py - radius), min(ih, py + radius + 1)
        patch = img[y_lo:y_hi, x_lo:x_hi].astype(np.int16)
        # Per-pixel max channel-diff vs substrate
        per_pixel_diff = np.abs(patch - substrate_rgb).max(axis=2)
        max_diff = int(per_pixel_diff.max())
        # If every pixel in the neighborhood reads as substrate (diff <= ~25
        # tolerates anti-aliasing + the 0.5 alpha jitter), the projected
        # center has nothing drawn there -- the render disagrees with the
        # file.
        if max_diff <= 25:
            failures.append(
                f"  {ref}: file=(({x_mm:.2f},{y_mm:.2f})mm) -> pixel "
                f"({px},{py}) reads pure-substrate (max channel diff "
                f"{max_diff})"
            )
        checked += 1

    assert checked > 0, "no footprints sampled -- fixture or AABB broken"
    assert not failures, (
        f"{len(failures)}/{checked} footprints projected to pure-substrate "
        f"pixels in GUI render -- coordinate handling drifted:\n"
        + "\n".join(failures)
    )
