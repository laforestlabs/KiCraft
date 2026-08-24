"""Curated example design briefs for the web app's onboarding.

These drive the animated cycling placeholder on the landing page (``web.index``):
it types through ``EXAMPLE_PROMPTS`` as passive inspiration.

(The "Surprise me" button no longer draws from this list — it now streams the
vetted self-eval corpus, ``kicraft.tuning.benchmark.BENCHMARK_PROMPTS``, in order
and actually *runs* each brief, so repeated clicks produce a continuous feed of
known-good designs — including the non-rectangular shaped boards (circle, hexagon,
star, snowman, …) now folded into that corpus. See ``web.index``'s ``surprise``
handler.)

The list is deliberately ambitious-but-proven: every entry is shaped like a design
the web pipeline completes well (ESP32-S3 sensor boards, USB-C power, LED drivers,
discrete sensor breakouts), so a bold ask ends in a finished board rather than a
stalled run. It is plain data so it stays easy to curate.
"""
from __future__ import annotations

# Full briefs the animated placeholder cycles through.
EXAMPLE_PROMPTS = [
    "An ESP32-S3 HUB75 display controller with USB-C PD 5 V input, an addressable LED output, and a small speaker output.",
    "An ESP32-S3 robot controller with two DRV8833 motor drivers, a buck regulator from a 2S battery, and motor screw terminals.",
    "A CAN bus node: an STM32 MCU, an SN65HVD230 transceiver, a DB9 connector, and a switchable bus terminator.",
    "A 1 A constant-current driver for a single power LED, USB-C input, with a heatsink copper area, no microcontroller.",
]
