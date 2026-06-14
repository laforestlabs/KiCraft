"""Curated, prebuilt KiCraft sample projects for the web showcase.

Single source of truth, reused two ways (mirrors how ``examples.py`` feeds the
prompt box): the public landing page (``web._render_landing``) shows an
interactive 3D model of each board (a ``<model-viewer>`` of ``board.glb``, with the
rendered ``board.png`` as poster / no-JS fallback) and the brief that produced it,
and the logged-in explorer (``web.samples_page``) opens the real KiCad files in
KiCanvas.

Every sample is a *finished* KiCraft output: a hierarchical schematic, real parts,
and a placed-and-routed board, curated from a real build into
``kicraft/server/sample_projects/<id>/``. The files are static assets, so showing
them costs zero tokens: the public page can showcase real work without ever calling
a model. (No model usage before a valid email signup is a hard product rule, so the
showcase is deliberately prebuilt rather than generated on demand.)

To add a sample, run ``scripts/promote_to_sample.py`` on a finished self-eval run:
it copies the curated KiCad files under ``sample_projects/<new-id>/`` (root + leaf
``*.kicad_sch``, ``*.kicad_pcb``, ``*.kicad_pro``; no ``.experiments``), stages the
bundle 3D models under ``3dmodels/``, renders ``previews/board.png`` and exports the
interactive ``previews/board.glb`` with ``kicad-cli``, and prints the ``Sample`` to
append below. The landing and explorer render whatever ``available_samples()``
returns, so a partially-synced deploy degrades to fewer cards rather than broken
images; ``board.glb`` is optional — a sample without it falls back to the PNG.
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

    @staticmethod
    def _cache_bust(path: Path) -> str:
        """``?v=<mtime>`` so a regenerated preview busts the browser cache. The
        /samples static mount serves these with ``Cache-Control: max-age=3600``
        at a stable URL, so without this a refreshed board.glb / render keeps
        showing the old cached bytes for up to an hour."""
        try:
            return f"?v={int(path.stat().st_mtime)}"
        except OSError:
            return ""

    @property
    def board_png_url(self) -> str:
        return f"{SAMPLES_URL}/{self.id}/previews/board.png{self._cache_bust(self.board_png)}"

    @property
    def board_hero(self) -> Path:
        """A dedicated, polished hero render (perspective + floor shadow, larger)
        for the featured board's hero. Optional — falls back to board.png."""
        return self.dir / "previews" / "hero.png"

    @property
    def board_hero_url(self) -> str:
        """URL for the hero image: the dedicated hero render if bundled, else the
        standard board render."""
        if self.board_hero.is_file():
            return (f"{SAMPLES_URL}/{self.id}/previews/hero.png"
                    f"{self._cache_bust(self.board_hero)}")
        return self.board_png_url

    @property
    def board_glb(self) -> Path:
        return self.dir / "previews" / "board.glb"

    @property
    def board_glb_url(self) -> str:
        return f"{SAMPLES_URL}/{self.id}/previews/board.glb{self._cache_bust(self.board_glb)}"

    def has_3d(self) -> bool:
        """True when an interactive GLB model is bundled (else use the PNG)."""
        return self.board_glb.is_file()

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
        id="usb-pd-trigger",
        title="USB-C PD trigger",
        blurb="A USB-C Power Delivery trigger that outputs a switch-selectable "
              "9 V, 12 V, or 20 V.",
        prompt="A USB-C PD trigger board that outputs a switch-selectable 9 V, "
               "12 V, or 20 V.",
        stem="USB_PD_TRIGGER",
        sheets=2, parts=12, featured=True,
    ),
    Sample(
        id="weather-sensor",
        title="BMP280 weather sensor",
        blurb="A barometric pressure and temperature sensor on a Qwiic/STEMMA bus, "
              "USB-C powered.",
        prompt="A BMP280 barometric weather sensor on a Qwiic/STEMMA bus, USB-C.",
        stem="BMP280_QWIIC_USB_C",
        sheets=3, parts=13,
    ),
    Sample(
        id="bench-breakout",
        title="USB-C bench breakout",
        blurb="Bench power from USB-C: regulated 3.3 V and 5 V rails with status "
              "LEDs.",
        prompt="A bench breakout: USB-C in, regulated 3.3 V and 5 V rails with "
               "status LEDs.",
        stem="BENCH_BREAKOUT",
        sheets=5, parts=19,
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
