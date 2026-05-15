"""Single PCB-view preset registry shared by every render consumer.

Each entry names a layer set + monitor-style post settings. Consumers
import ``VIEWS`` from here instead of declaring their own (the GUI
monitor, the CLI render command, the scoring visual check, the
subcircuit diagnostics bundle, and the parent compose stamper all
flow through this one dict). Adding a view here makes it available
to every consumer with zero further wiring.
"""

from __future__ import annotations


VIEWS: dict[str, dict] = {
    "front_all": {
        # Top-down PCBnew-like view. F.Cu+B.Cu together triggers the
        # composite path in the renderer (B.Cu rendered at reduced
        # opacity so it doesn't obscure front detail). F.Mask is
        # intentionally OMITTED: KiCad renders it as an opaque
        # solder-mask-colored fill over the whole board, which would
        # obscure B.Cu and produce a "blank blue PCB" when the router
        # put traces on the back layer.
        #
        # Post settings deliberately gentle (contrast 1.15, sat 1.05):
        # the heavier boost used elsewhere washed out B.Cu against
        # the saturated background.
        "layers": "B.Cu,F.Cu,F.SilkS,Edge.Cuts",
        "desc": "Top-down view: both copper layers + silkscreen + outline (PCBnew-like)",
        "post": {
            "contrast": 1.15,
            "saturation": 1.05,
            "brightness": 1.00,
        },
    },
    "back_all": {
        "layers": "B.Cu,B.SilkS,B.Mask,Edge.Cuts",
        "desc": "Back copper + silkscreen + mask + outline",
        "mirror": True,
        "post": {
            "contrast": 1.38,
            "saturation": 1.24,
            "brightness": 0.90,
        },
    },
    "copper_both": {
        "layers": "F.Cu,B.Cu,Edge.Cuts",
        "desc": "Both copper layers + outline",
        "post": {
            "contrast": 1.34,
            "saturation": 1.18,
            "brightness": 0.90,
        },
    },
    "front_copper": {
        "layers": "F.Cu,Edge.Cuts",
        "desc": "Front copper traces and pads only",
        "post": {
            "contrast": 1.30,
            "saturation": 1.12,
            "brightness": 0.90,
        },
    },
    "back_copper": {
        "layers": "B.Cu,Edge.Cuts",
        "desc": "Back copper (ground plane, traces)",
        "mirror": True,
        "post": {
            "contrast": 1.30,
            "saturation": 1.12,
            "brightness": 0.90,
        },
    },
    "courtyard": {
        "layers": "F.CrtYd,B.CrtYd,Edge.Cuts",
        "desc": "Component courtyards for overlap review",
        "post": {
            "contrast": 1.34,
            "saturation": 1.02,
            "brightness": 0.90,
        },
    },
}
