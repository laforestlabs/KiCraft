"""Curated example design briefs for the web app's onboarding.

These drive the animated cycling placeholder on the landing page (``web.index``):
it types through ``EXAMPLE_PROMPTS`` as passive inspiration.

(The "Surprise me" button no longer draws from this list — it now streams the
vetted self-eval corpus, ``kicraft.tuning.benchmark.BENCHMARK_PROMPTS``, in order
and actually *runs* each brief, so repeated clicks produce a continuous feed of
known-good designs. See ``web.index``'s ``surprise`` handler.)

The list is deliberately ambitious-but-proven: every entry is shaped like a design
the web pipeline completes well (ESP32-S3 sensor boards, USB-C power, LED drivers,
discrete sensor breakouts), so a bold ask ends in a finished board rather than a
stalled run. It is plain data so it stays easy to curate.
"""
from __future__ import annotations

# Full briefs the animated placeholder cycles through.
EXAMPLE_PROMPTS = [
    "An ESP32-S3 plant monitor: soil moisture + BME280 temp/humidity, a small OLED, USB-C power.",
    "A USB-C PD trigger board that outputs a switch-selectable 9 V, 12 V, or 20 V.",
    "An 8-channel addressable WS2812 LED driver running off 5 V USB-C.",
    "A BMP280 barometric weather sensor on a Qwiic/STEMMA bus, USB-C.",
    "A USB-C rechargeable 18650 flashlight: boost driver, three white LEDs, no microcontroller.",
    "An ESP32-S3 automatic cat feeder: stepper-driven auger, load-cell food sensor, USB-C.",
    "A bench breakout: USB-C in, regulated 3.3 V and 5 V rails with status LEDs.",
    "An ESP32-S3 environmental logger: BME280 + ambient light sensor + microSD, USB-C.",
    "A motion-activated USB-C night light with a PIR sensor and warm-white LEDs.",
]
