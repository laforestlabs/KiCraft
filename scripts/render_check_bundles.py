"""Visual render review of parts-library bundles, for human eyeballing.

For each bundle (explicit paths or --all-vendored): build a one-footprint
board with an Edge.Cuts margin around the part, copy the bundle's 3d/
models to <board dir>/3dmodels/<name>/ (mirroring what synthesis stages
into generated projects), then `kicad-cli pcb render` a top and an oblique
view. The PNGs land under --out (default .render-check/, gitignored) for
review before `promote-part --to production`.

Catches what the validators cannot: a model that resolves but sits
rotated, offset, or on the wrong side of its footprint (the alignment bug
class found on esp32-wroom-32e-n4 during the 3D backfill).

Run with the repo venv (needs pcbnew): .venv/bin/python scripts/render_check_bundles.py --all-vendored
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pcbnew

REPO = Path(__file__).resolve().parent.parent

VIEWS = {
    "top": ["--side", "top"],
    "oblique": ["--rotate", "-35,0,25", "--zoom", "0.85"],
}


def render_bundle(part_dir: Path, out: Path) -> list[str]:
    """Render one bundle; returns a list of failure strings (empty = ok)."""
    name = part_dir.name
    prettys = sorted(part_dir.glob("*.pretty"))
    if not prettys:
        return [f"{name}: no .pretty dir"]
    mods = sorted(prettys[0].glob("*.kicad_mod"))
    if not mods:
        return [f"{name}: no .kicad_mod in {prettys[0].name}"]

    work = out / "_boards" / name
    work.mkdir(parents=True, exist_ok=True)
    model_src = part_dir / "3d"
    if model_src.is_dir():
        shutil.copytree(model_src, work / "3dmodels" / name, dirs_exist_ok=True)

    board_path = work / f"{name}.kicad_pcb"
    board = pcbnew.NewBoard(str(board_path))
    fp = pcbnew.FootprintLoad(str(prettys[0]), mods[0].stem)
    if fp is None:
        return [f"{name}: FootprintLoad returned None for {mods[0].stem}"]
    board.Add(fp)
    fp.SetPosition(pcbnew.VECTOR2I(0, 0))
    bbox = fp.GetBoundingBox()
    margin = pcbnew.FromMM(2)
    rect = pcbnew.PCB_SHAPE(board)
    rect.SetShape(pcbnew.SHAPE_T_RECT)
    rect.SetStart(pcbnew.VECTOR2I(bbox.GetLeft() - margin, bbox.GetTop() - margin))
    rect.SetEnd(pcbnew.VECTOR2I(bbox.GetRight() + margin, bbox.GetBottom() + margin))
    rect.SetLayer(pcbnew.Edge_Cuts)
    rect.SetWidth(pcbnew.FromMM(0.1))
    board.Add(rect)
    board.Save(str(board_path))

    failures = []
    for view, extra in VIEWS.items():
        r = subprocess.run(
            ["kicad-cli", "pcb", "render",
             "-o", str(out / f"{name}_{view}.png"),
             "--quality", "high", "--background", "opaque",
             "-w", "640", "-h", "480", *extra, str(board_path)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            failures.append(f"{name} ({view}): {r.stderr.strip()[:200]}")
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="bundle directories to render")
    ap.add_argument("--all-vendored", action="store_true",
                    help="render every bundle in kicraft/parts_library/")
    ap.add_argument("--out", default=str(REPO / ".render-check"),
                    help="output dir for the PNGs (default: .render-check/)")
    args = ap.parse_args(argv)

    if args.all_vendored:
        base = REPO / "kicraft" / "parts_library"
        part_dirs = sorted(d for d in base.iterdir()
                           if d.is_dir() and (d / "manifest.json").is_file())
    else:
        part_dirs = [Path(p).resolve() for p in args.paths]
    if not part_dirs:
        ap.error("no bundles (pass paths or --all-vendored)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok, failures = [], []
    for part_dir in part_dirs:
        errs = render_bundle(part_dir, out)
        if errs:
            failures += errs
        else:
            ok.append(part_dir.name)

    print(f"rendered {len(ok)}: {' '.join(ok)}")
    print(f"review the PNGs in {out}/")
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
