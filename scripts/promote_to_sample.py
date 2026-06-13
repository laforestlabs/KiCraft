"""Promote a finished self-eval board into a landing-page showcase sample.

Given a self-eval run dir (``logs/self_eval/<batch>/run_NN_.../``), this curates
its routed board into ``kicraft/server/sample_projects/<id>/`` exactly the way
``samples.py`` documents, plus the 3D assets the new landing page needs:

1. copy the root + leaf ``*.kicad_sch``, the routed ``<stem>.kicad_pcb`` and
   ``<stem>.kicad_pro`` (no ``.experiments``, ``fab/``, ``_best`` or reports);
2. stage every bundle library's ``3d/`` models into ``3dmodels/<lib>/`` and
   **rewrite the board's broken bundle model refs** (routed boards emit bundle
   models as a bare ``/X.wrl`` absolute path that resolves nowhere) to
   ``${KIPRJMOD}/3dmodels/<lib>/X.wrl`` so kicad-cli resolves them. Stock
   ``${KICAD9_3DMODEL_DIR}/...`` refs already resolve and are left alone;
3. render ``previews/board.png`` (assembled 3D, the poster / og:image / no-JS
   fallback) and export ``previews/board.glb`` (the interactive model the hero
   and gallery rotate via <model-viewer>).

Then it prints a ready-to-paste ``Sample(...)`` entry (computed sheets/parts,
the brief as the prompt). Curate id/title/blurb; the prompt defaults to the
run's ``brief.txt``.

Run with the repo venv (kicad-cli + xvfb-run on PATH):
    .venv/bin/python scripts/promote_to_sample.py logs/self_eval/<b>/run_04_* \\
        --id bmp280-weather --title "BMP280 weather sensor" --blurb "..." --featured
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from kicraft.design.synthesis.models3d import stage_3d_models  # noqa: E402

SAMPLES_DIR = REPO / "kicraft" / "server" / "sample_projects"
_MODEL_RE = re.compile(r'\(model "([^"]+)"')


def _run(cmd: list[str], timeout: float = 300) -> None:
    """kicad-cli, retrying under xvfb-run when the box has no GL context."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode == 0:
        return
    if shutil.which("xvfb-run"):
        r = subprocess.run(["xvfb-run", "-a", *cmd], capture_output=True,
                           text=True, timeout=timeout)
        if r.returncode == 0:
            return
    raise RuntimeError(f"{' '.join(cmd[:4])} failed (rc={r.returncode}): "
                       f"{(r.stderr or r.stdout).strip()[:300]}")


def _find_generated(run_dir: Path) -> tuple[Path, str]:
    """(project_dir, stem) for the single generated project under run_dir."""
    state = run_dir / ".kicraft" / "state.json"
    stem = json.loads(state.read_text())["project_stem"] if state.is_file() else ""
    gen = run_dir / "generated"
    if stem and (gen / stem).is_dir():
        return gen / stem, stem
    # Fallback: the lone subdir, stem from its root .kicad_pcb (skip *_best).
    subdirs = [d for d in gen.iterdir() if d.is_dir()] if gen.is_dir() else []
    if len(subdirs) != 1:
        raise SystemExit(f"expected one project under {gen}, found {len(subdirs)}")
    proj = subdirs[0]
    pcbs = [p for p in proj.glob("*.kicad_pcb") if not p.stem.endswith("_best")]
    if not pcbs:
        raise SystemExit(f"no routed .kicad_pcb in {proj}")
    return proj, pcbs[0].stem


def _stage_and_fix_models(proj: Path, dest: Path, stem: str,
                          bom_parts: list[dict]) -> int:
    """Stage bundle models into dest/3dmodels and rewrite the board's refs."""
    bom = SimpleNamespace(parts=[SimpleNamespace(footprint=p.get("footprint", ""))
                                 for p in bom_parts])
    stage_3d_models(dest, bom, project_root=REPO)
    # basename -> ${KIPRJMOD}/3dmodels/<lib>/<basename> for every staged model.
    staged = {f.name: "${KIPRJMOD}/" + str(f.relative_to(dest))
              for f in sorted(dest.glob("3dmodels/*/*")) if f.is_file()}
    pcb = dest / f"{stem}.kicad_pcb"
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        path = m.group(1)
        if path.startswith("${KICAD9_3DMODEL_DIR}"):
            return m.group(0)  # stock model, resolves through system KiCad
        base = path.rsplit("/", 1)[-1]
        if base in staged and path != staged[base]:
            n += 1
            return f'(model "{staged[base]}"'
        return m.group(0)

    pcb.write_text(_MODEL_RE.sub(repl, pcb.read_text()))
    return n


def promote(run_dir: Path, sample_id: str, *, rotate: str, zoom: str,
            width: int, height: int, background: str = "transparent") -> dict:
    proj, stem = _find_generated(run_dir)
    state = run_dir / ".kicraft" / "state.json"
    bom_parts = (json.loads(state.read_text()).get("bom", {}).get("parts", [])
                 if state.is_file() else [])

    dest = SAMPLES_DIR / sample_id
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "previews").mkdir(parents=True)

    # Curated KiCad tree only: schematics, the routed board, the project file.
    for sch in proj.glob("*.kicad_sch"):
        shutil.copy2(sch, dest / sch.name)
    shutil.copy2(proj / f"{stem}.kicad_pcb", dest / f"{stem}.kicad_pcb")
    pro = proj / f"{stem}.kicad_pro"
    if pro.is_file():
        shutil.copy2(pro, dest / pro.name)

    n_fixed = _stage_and_fix_models(proj, dest, stem, bom_parts)

    board = str(dest / f"{stem}.kicad_pcb")
    kiprjmod = ["-D", f"KIPRJMOD={dest}"]
    # Transparent so the board floats on the page (poster matches <model-viewer>'s
    # transparent canvas; the CSS drop-shadow gives the lift).
    _run(["kicad-cli", "pcb", "render", "-o", str(dest / "previews" / "board.png"),
          "--quality", "high", "--background", background, "--rotate", rotate,
          "--zoom", zoom, "-w", str(width), "-h", str(height), *kiprjmod, board])
    _run(["kicad-cli", "pcb", "export", "glb", "--subst-models", "--force",
          *kiprjmod, "-o", str(dest / "previews" / "board.glb"), board])

    sheets = len(list(dest.glob("*.kicad_sch")))
    parts = len(bom_parts) or \
        (dest / f"{stem}.kicad_pcb").read_text().count("(footprint ")
    return {"dest": dest, "stem": stem, "sheets": sheets, "parts": parts,
            "models_fixed": n_fixed,
            "glb_kb": (dest / "previews" / "board.glb").stat().st_size // 1024,
            "png_kb": (dest / "previews" / "board.png").stat().st_size // 1024}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path, help="self-eval run dir (run_NN_...)")
    ap.add_argument("--id", required=True, help="URL slug / sample dir name")
    ap.add_argument("--title", default="", help="display title")
    ap.add_argument("--blurb", default="", help="one-line description")
    ap.add_argument("--prompt", default="", help="brief (default: run's brief.txt)")
    ap.add_argument("--featured", action="store_true", help="the hero sample")
    ap.add_argument("--rotate", default="-58,0,28", help="kicad-cli render --rotate")
    ap.add_argument("--zoom", default="0.95", help="kicad-cli render --zoom")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--background", default="transparent",
                    help="render bg: transparent (default), opaque, default")
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    info = promote(run_dir, args.id, rotate=args.rotate, zoom=args.zoom,
                   width=args.width, height=args.height, background=args.background)

    prompt = args.prompt
    if not prompt:
        bt = run_dir / "brief.txt"
        prompt = bt.read_text().strip() if bt.is_file() else ""

    print(f"\npromoted -> {info['dest']}")
    print(f"  stem={info['stem']} sheets={info['sheets']} parts={info['parts']} "
          f"models_rewritten={info['models_fixed']} "
          f"png={info['png_kb']}KB glb={info['glb_kb']}KB")
    print("\n--- paste into samples.py SAMPLES ---")
    feat = "\n        featured=True," if args.featured else ""
    print(f'''    Sample(
        id="{args.id}",
        title="{args.title}",
        blurb="{args.blurb}",
        prompt="{prompt}",
        stem="{info['stem']}",
        sheets={info['sheets']}, parts={info['parts']},{feat}
    ),''')
    return 0


if __name__ == "__main__":
    sys.exit(main())
