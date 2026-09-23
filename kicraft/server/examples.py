"""Onboarding briefs for the web app: the animated placeholder and "Surprise me".

``EXAMPLE_PROMPTS`` drives the animated cycling placeholder on the landing page
(``web.index``): it types through them as passive inspiration.

The "Surprise me" button does NOT replay a saved brief. Each click composes a
fresh one with ``generate_brief(seed)`` from a persistent seed counter, so no two
clicks repeat and any single brief is reproducible from its seed. It draws from
neither ``kicraft.tuning.benchmark.BENCHMARK_PROMPTS`` (the 34-brief self-eval /
tuning corpus) nor ``EXAMPLE_PROMPTS``. See ``web.index``'s ``surprise`` handler.

The generated space is deliberately bounded by ``BRIEF_SLOTS``: every brief stays
inside the envelope the eval campaigns froze (USB or up to 24 V DC, up to 3 A,
two or four layers, at most 100 x 100 mm, module parts allowed) and names only
common, sourceable parts, so a surprise is a reasonable ask rather than an
unbounded one. It is plain data plus a pure function, so it stays easy to curate
and easy to test.

Inspect the generated stream (seeds are reproducible):

    python -m kicraft.server.examples 20
"""
from __future__ import annotations

import random
import re
import string

# Full briefs the animated placeholder cycles through.
EXAMPLE_PROMPTS = [
    "An ESP32-S3 HUB75 display controller with USB-C PD 5 V input, an addressable LED output, and a small speaker output.",
    "An ESP32-S3 robot controller with two DRV8833 motor drivers, a buck regulator from a 2S battery, and motor screw terminals.",
    "A CAN bus node: an STM32 MCU, an SN65HVD230 transceiver, a DB9 connector, and a switchable bus terminator.",
    "A 1 A constant-current driver for a single power LED, USB-C input, with a heatsink copper area, no microcontroller.",
]

# Seed range the persistent Surprise-me counter wraps at. Only the counter is
# bounded (it is persisted as a plain setting string); the seed space the
# generator itself accepts is unbounded.
BRIEF_SEED_MODULO = 2**31

# Every value a template may interpolate. Keys are the field names used in
# ``BRIEF_TEMPLATES``; values are bare noun phrases (no article -- ``{name:a}``
# adds the right one). Ratings and sizes stay inside the frozen design envelope
# (<=24 V, <=3 A, <=100 x 100 mm).
BRIEF_SLOTS: dict[str, tuple[str, ...]] = {
    "mcu": (
        "ESP32-C3 module",
        "ESP32-S3 module",
        "RP2040",
        "STM32G0",
        "ATtiny1604",
        "CH32V003",
    ),
    "supply": (
        "USB-C 5 V input",
        "12 V DC barrel jack",
        "24 V DC screw terminal",
        "2S Li-ion battery pack",
        "5 V header from the host board",
        "18 V DC input",
    ),
    "bus": (
        "Qwiic/STEMMA QT I2C connector",
        "I2C header",
        "SPI header",
        "UART header",
    ),
    "io": (
        "four screw terminals",
        "a 6-pin 0.1 inch header",
        "two JST-XH connectors",
        "an 8-pin 0.1 inch header",
        "three push buttons",
        "a 4-pin JST connector",
    ),
    "indicator": (
        "power LED",
        "status LED",
        "RGB status LED",
        "secondary status LED",
    ),
    "sensor": (
        "temperature",
        "humidity",
        "ambient-light",
        "current-sense",
        "hall-effect",
    ),
    "driver": (
        "DRV8833 dual H-bridge",
        "A4988 stepper driver socket",
        "ULN2003 darlington array",
        "TB6612FNG dual H-bridge",
    ),
    "led": (
        "WS2812 addressable LED strip",
        "3 W power LED",
        "4-digit 7-segment display",
        "LED matrix",
    ),
    "opamp": ("MCP6002", "LM358", "TLV9062"),
    "bridge": ("CH340C USB-UART bridge", "CP2102N USB-UART bridge"),
    "rail": ("3.3 V", "5 V", "12 V"),
    "amps": ("0.5", "1", "2", "3"),
    "channels": ("two", "four", "eight"),
    "relay_n": ("two", "three", "four"),
    "size": (
        "30 x 20 mm",
        "40 x 30 mm",
        "50 x 50 mm",
        "60 x 40 mm",
        "80 x 60 mm",
        "100 x 100 mm",
    ),
}

# (family, templates). A template is one user-shaped request. ``{name}`` draws a
# value from ``BRIEF_SLOTS`` (reused if the name appears more than once) and
# ``{name:a}`` renders it with the correct indefinite article.
BRIEF_TEMPLATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "controller",
        (
            "{mcu:a} controller board powered from {supply:a}, with {io} for I/O, {bus:a}, and {indicator:a}.",
            "{mcu:a} USB-serial bridge board: {supply:a}, {bridge:a}, {io}, and {indicator:a}.",
            "{mcu:a} development board with {supply:a}, {bus:a}, {io}, a reset button, and {indicator:a}.",
            "{mcu:a} interface board that adapts {supply:a} to {rail} logic, with {io} and {bus:a}.",
        ),
    ),
    (
        "sensors",
        (
            "{sensor:a} sensor breakout with {mcu:a}, {supply:a}, {bus:a}, and {io}.",
            "A data logger with {channels} {sensor} channels using {mcu:a}, powered from {supply:a}, with {bus:a} and {io}.",
            "A battery-powered {sensor} node: {mcu:a}, {supply:a}, {bus:a}, {io}, and {indicator:a}.",
            "{sensor:a} acquisition board with {mcu:a}, {supply:a}, {io}, kept under {size}.",
        ),
    ),
    (
        "display_led",
        (
            "{mcu:a} LED driver board for {led:a}, powered from {supply:a}, with {bus:a} and {io}.",
            "An indicator controller with {channels} outputs, using {mcu:a}, {supply:a}, {led:a}, and {indicator:a}.",
            "A small {led} display board: {mcu:a}, {supply:a}, {bus:a}, and {io}.",
        ),
    ),
    (
        "power",
        (
            "{supply:a} to {rail} regulator board rated {amps} A with {io}, a power LED, and an enable jumper.",
            "{supply:a} power distribution board with {channels} fused outputs on screw terminals and {indicator:a}.",
            "{supply:a} to {rail} converter with reverse-polarity protection, {io}, and a power LED.",
        ),
    ),
    (
        "analog",
        (
            "{opamp:a}-based {sensor} amplifier board with {supply:a}, {io}, and a trim potentiometer.",
            "A front end with {channels} {sensor} channels using {opamp:a}, supplied from {supply:a}, with {io}.",
            "A precision {sensor} conditioning board: {opamp}, {supply:a}, {io}, and {indicator:a}, under {size}.",
        ),
    ),
    (
        "actuator",
        (
            "{mcu:a} motor controller with {driver:a}, {supply:a}, and motor screw terminals.",
            "{mcu:a} relay board with {relay_n} relay outputs on screw terminals, {supply:a}, and {indicator:a}.",
            "{mcu:a} actuator driver: {driver:a}, {supply:a}, {io}, and {indicator:a}.",
        ),
    ),
)

# One extra requirement appended to every brief. Besides making each request
# read like a real one, this multiplies the composition space the way a lone
# template cannot -- two clicks must not hand out the same ask.
BRIEF_TRAILING = (
    "Keep it under {size}.",
    "Use a two-layer stack-up.",
    "Use a four-layer stack-up.",
    "Make the input reverse-polarity protected.",
    "Add an enable jumper.",
    "Keep it under {size} and use common, easily sourced parts.",
    "Put all the connectors on one edge.",
    "Add a reset button.",
)

# Tokens whose spoken form contradicts the naive first-letter rule ("a USB-C"
# not "an USB-C": the letter is pronounced "you"). Only the genuinely ambiguous
# ones live here; everything else falls through to the vowel test.
_ARTICLE_FOR_TOKEN = {
    "RP2040": "an",
    "STM32G0": "an",
    "USB-C": "a",
    "UART": "a",
    "SPI": "an",
    "RGB": "an",
    "LED": "an",
    "MCP6002": "an",
    "LM358": "an",
    "ULN2003": "a",
}

_VOWELS = frozenset("aeiou")


def indefinite(phrase: str) -> str:
    """``phrase`` prefixed with "a"/"an", by spoken form for the known acronyms
    in ``BRIEF_SLOTS`` and by the ordinary vowel rule otherwise."""
    token = phrase.split()[0]
    article = _ARTICLE_FOR_TOKEN.get(token)
    if article is None and token[:1].isdigit():
        digits = re.match(r"\d+", token).group(0)
        # "an 8", "an 11", "an 18", "an 80" (the leading number is spelled with
        # a vowel sound); every other leading number takes "a".
        article = "an" if digits[0] == "8" or digits in ("11", "18") else "a"
    if article is None:
        article = "an" if token[:1].lower() in _VOWELS else "a"
    return f"{article} {phrase}"


class _Slots(string.Formatter):
    """``str.format`` with two extras: ``{name}`` resolves a lazily drawn slot
    from ``BRIEF_SLOTS`` (cached, so a repeated field is consistent within one
    brief) and ``{name:a}`` renders it with the indefinite article."""

    def __init__(self, rng: random.Random):
        super().__init__()
        self._rng = rng
        self._drawn: dict[str, str] = {}

    def get_value(self, key, args, kwargs):
        if isinstance(key, int):
            return args[key]
        if key not in self._drawn:
            self._drawn[key] = self._rng.choice(BRIEF_SLOTS[key])
        return self._drawn[key]

    def format_field(self, value, format_spec):
        if format_spec == "a":
            return indefinite(value)
        return super().format_field(value, format_spec)


def generate_brief(seed: int) -> str:
    """One brief, a pure function of ``seed``.

    The same seed always yields the same brief, so a surprising run can be
    reproduced afterwards; distinct seeds are the Surprise-me button's variety.
    """
    rng = random.Random(int(seed))
    _family, templates = rng.choice(BRIEF_TEMPLATES)
    slots = _Slots(rng)
    text = slots.vformat(rng.choice(templates), (), {})
    text = f"{text} {slots.vformat(rng.choice(BRIEF_TRAILING), (), {})}"
    return text[:1].upper() + text[1:]


if __name__ == "__main__":
    import sys

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for i in range(count):
        print(f"[{i}] {generate_brief(i)}")
