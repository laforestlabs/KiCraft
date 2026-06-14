"""Regenerate the showcase samples' 3D previews in place (no re-promote).

The landing page bundles each sample's ``previews/board.glb`` (interactive
<model-viewer>) and ``previews/board.png`` (poster / og:image), plus a polished
``previews/hero.png`` for the featured board's hero. Older bundles shipped GLBs
exported without the board's copper / silkscreen / soldermask layers (a blank
white slab) and with a few bare ``/X.wrl`` component refs that resolved nowhere
(missing part bodies).

This re-exports those assets against the *exact* boards already curated under
``kicraft/server/sample_projects/<id>/`` — it never touches the .kicad_pcb
geometry — reusing the hardened helpers in :mod:`promote_to_sample`:

* resolve every component model ref (staging fetched/library bodies as needed),
* re-export ``board.glb`` with the realistic board layers, and
* render a dedicated ``hero.png`` for the featured sample.

Run with the repo venv (kicad-cli + xvfb-run on PATH):
    .venv/bin/python scripts/refresh_sample_previews.py            # all samples
    .venv/bin/python scripts/refresh_sample_previews.py weather-sensor
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from kicraft.server.samples import SAMPLES  # noqa: E402
from scripts.promote_to_sample import (  # noqa: E402
    SAMPLES_DIR,
    export_board_glb,
    render_board_png,
    _resolve_model_refs,
)

# A flat, mostly-top-down isometric with a little tilt to show component height,
# grounded with a floor shadow for depth. Every landing still uses this look; the
# featured board also gets a larger hero.png.
ISO = {"rotate": "-24,0,16", "zoom": "0.86", "floor": True, "perspective": True}
BOARD = {**ISO, "width": 1600, "height": 1200}
HERO = {**ISO, "width": 2200, "height": 1650}


def refresh(sample) -> dict:
    dest = SAMPLES_DIR / sample.id
    stem = sample.stem
    if not (dest / f"{stem}.kicad_pcb").is_file():
        raise SystemExit(f"no board at {dest}/{stem}.kicad_pcb")

    n_fixed, unresolved = _resolve_model_refs(dest, stem)
    n_recolored = export_board_glb(dest, stem)
    render_board_png(dest, stem, out=dest / "previews" / "board.png", **BOARD)
    if sample.featured:
        render_board_png(dest, stem, out=dest / "previews" / "hero.png", **HERO)

    glb_kb = (dest / "previews" / "board.glb").stat().st_size // 1024
    return {"id": sample.id, "fixed": n_fixed, "unresolved": unresolved,
            "recolored": n_recolored, "glb_kb": glb_kb, "hero": sample.featured}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ids", nargs="*", help="sample id(s); default: all")
    args = ap.parse_args(argv)

    samples = [s for s in SAMPLES if not args.ids or s.id in args.ids]
    if not samples:
        raise SystemExit(f"no matching samples (known: {[s.id for s in SAMPLES]})")

    rc = 0
    for s in samples:
        info = refresh(s)
        tail = " +hero.png" if info["hero"] else ""
        print(f"{info['id']}: refs_fixed={info['fixed']} "
              f"recolored={info['recolored']} glb={info['glb_kb']}KB{tail}")
        if info["unresolved"]:
            rc = 1
            print(f"  WARNING unresolved: {', '.join(sorted(set(info['unresolved'])))}",
                  file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
