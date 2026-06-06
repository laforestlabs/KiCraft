"""Curated, prebuilt KiCraft sample projects for the web showcase.

Single source of truth, reused two ways (mirrors how ``examples.py`` feeds the
prompt box): the public landing page (``web._render_landing``) shows each sample's
3D board render and the brief that produced it, and the logged-in explorer
(``web.samples_page``) opens the real KiCad files in KiCanvas.

Every sample is a *finished* KiCraft output: a hierarchical schematic, real parts,
and a placed-and-routed board, curated from a real build into
``kicraft/server/sample_projects/<id>/``. The files are static assets, so showing
them costs zero tokens: the public page can showcase real work without ever calling
a model. (No model usage before a valid email signup is a hard product rule, so the
showcase is deliberately prebuilt rather than generated on demand.)

To add a sample: drop its curated KiCad files under ``sample_projects/<new-id>/``
(root + leaf ``*.kicad_sch``, ``*.kicad_pcb``, ``*.kicad_pro``; no ``.experiments``),
render ``previews/board.png`` with ``kicad-cli pcb render``, and append a ``Sample``
below. The landing and explorer render whatever ``available_samples()`` returns, so
a partially-synced deploy degrades to fewer cards rather than broken images.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SAMPLES_DIR: Path = Path(__file__).parent / "sample_projects"

# URL prefix the bundle is mounted at (see web.add_static_files). Preview images and
# the raw KiCad files KiCanvas fetches both live under here.
SAMPLES_URL = "/samples"


@dataclass(frozen=True)
class Sample:
    """One curated, finished design shown in the showcase.

    id      URL-safe slug; also the bundled directory name under SAMPLES_DIR.
    stem    project stem: the basename of the root .kicad_sch/.kicad_pcb/.kicad_pro.
    sheets  hierarchical sheet count (a display stat).
    parts   placed component count (a display stat).
    prompt  the brief that produced it; prefilled into the app on signup so a
            visitor can build their own version in one click (while logged in).
    featured  the single sample shown large in the hero.
    """

    id: str
    title: str
    blurb: str
    prompt: str
    stem: str
    sheets: int
    parts: int
    featured: bool = False

    @property
    def dir(self) -> Path:
        return SAMPLES_DIR / self.id

    @property
    def board_pcb(self) -> Path:
        return self.dir / f"{self.stem}.kicad_pcb"

    @property
    def board_png(self) -> Path:
        return self.dir / "previews" / "board.png"

    @property
    def board_png_url(self) -> str:
        return f"{SAMPLES_URL}/{self.id}/previews/board.png"

    def schematic_files(self) -> list[Path]:
        """The root schematic first, then leaf sheets (sorted), so KiCanvas links
        the hierarchy by filename the way the app's synth view expects."""
        root = f"{self.stem}.kicad_sch"
        schs = sorted(self.dir.glob("*.kicad_sch"),
                      key=lambda p: (p.name != root, p.name))
        return schs

    def schematic_sources(self) -> list[tuple[str, str]]:
        """(url, filename) for each schematic sheet, served from the static mount."""
        return [(f"{SAMPLES_URL}/{self.id}/{p.name}", p.name)
                for p in self.schematic_files()]

    def board_source(self) -> tuple[str, str]:
        """(url, filename) for the routed board, served from the static mount."""
        name = f"{self.stem}.kicad_pcb"
        return (f"{SAMPLES_URL}/{self.id}/{name}", name)

    def exists(self) -> bool:
        """True when both the routed board and its preview render are present."""
        return self.board_pcb.is_file() and self.board_png.is_file()


# Curated set. Prompts honestly describe each board, so building from a prefilled
# brief yields a design like the one shown.
SAMPLES: list[Sample] = [
    Sample(
        id="motion-sensor",
        title="ESP32-S3 motion sensor",
        blurb="A PIR motion controller: ESP32-S3, USB-C power, a 3.3 V regulator, "
              "and a programming header.",
        prompt="An ESP32-S3 PIR motion sensor board: USB-C power, a 3.3 V "
               "regulator, and a programming header.",
        stem="ESP32_MOTION_SENSOR",
        sheets=6, parts=22, featured=True,
    ),
    Sample(
        id="weather-sensor",
        title="BMP280 weather sensor",
        blurb="A barometric temperature and pressure sensor on a Qwiic/STEMMA bus, "
              "USB-C powered.",
        prompt="A BMP280 barometric weather sensor on a Qwiic/STEMMA bus, USB-C.",
        stem="USB_BMP280_READER",
        sheets=6, parts=17,
    ),
    Sample(
        id="led-matrix",
        title="ESP32 LED-matrix driver",
        blurb="An ambitious one: an ESP32-driven addressable LED matrix, 200+ "
              "placed parts, fully routed, USB-C powered.",
        prompt="An ESP32-driven addressable LED matrix board: USB-C power, a "
               "3.3 V rail, and a 5 V level shifter.",
        stem="ESP32_LED_MATRIX",
        sheets=6, parts=230,
    ),
]


def available_samples() -> list[Sample]:
    """Samples whose bundled board and preview are present on disk."""
    return [s for s in SAMPLES if s.exists()]


def get_sample(sample_id: str) -> Sample | None:
    """The Sample with this id, or None."""
    return next((s for s in SAMPLES if s.id == sample_id), None)


def featured_sample() -> Sample | None:
    """The hero sample: the one flagged ``featured`` (else the first available)."""
    avail = available_samples()
    return next((s for s in avail if s.featured), avail[0] if avail else None)
