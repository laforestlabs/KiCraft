"""Unified PCB rasterization.

One core function (``render_pcb``) renders a ``.kicad_pcb`` to a PNG
clipped to the board's Edge.Cuts AABB. A ``MonitorStyle`` adapter on
top fills the transparent pixels inside Edge.Cuts with the PCB
substrate color and applies a contrast/saturation boost for the
monitor / pipeline-graph views; without it the renderer produces a
transparent-background, exactly-Edge.Cuts-sized PNG that the manual
layout canvas drops directly onto its SVG. Both consumers go through
the same pipeline so they cannot drift.

``render_views`` is the multi-preset wrapper: pass a list of view
names from ``VIEWS`` and a filename template, get back a dict of
``{view: output_path}``. Every preset-based consumer (the CLI, the
score-time visual check, the subcircuit diagnostics bundle, the
parent compose stamper) goes through this one function.
"""

from kicraft.render.edge_cuts import parse_edge_cuts_aabb
from kicraft.render.pcb_renderer import (
    EdgeCutsExtent,
    MonitorStyle,
    render_pcb,
    render_views,
)
from kicraft.render.views import VIEWS

__all__ = [
    "EdgeCutsExtent",
    "MonitorStyle",
    "VIEWS",
    "parse_edge_cuts_aabb",
    "render_pcb",
    "render_views",
]
