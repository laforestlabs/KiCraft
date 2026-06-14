"""Promote a finished self-eval board into a landing-page showcase sample.

Given a self-eval run dir (``logs/self_eval/<batch>/run_NN_.../``), this curates
its routed board into ``kicraft/server/sample_projects/<id>/`` exactly the way
``samples.py`` documents, plus the 3D assets the new landing page needs:

1. copy the root + leaf ``*.kicad_sch``, the routed ``<stem>.kicad_pcb`` and
   ``<stem>.kicad_pro`` (no ``.experiments``, ``fab/``, ``_best`` or reports);
2. stage every bundle library's ``3d/`` models into ``3dmodels/<lib>/`` and
   **make every board model ref resolve** (routed boards emit bundle models as a
   bare ``/X.wrl`` absolute path that resolves nowhere): each is matched by
   basename against the staged models, the home parts cache and the vendored
   parts library, copying the model (+ STEP sibling) in and rewriting the ref to
   ``${KIPRJMOD}/3dmodels/<lib>/X.wrl``. Stock ``${KICAD9_3DMODEL_DIR}/...``
   refs already resolve and are left alone;
3. render ``previews/board.png`` (assembled 3D, the poster / og:image / no-JS
   fallback) and export ``previews/board.glb`` — the interactive model the
   gallery rotates via <model-viewer> — with the board's copper, silkscreen and
   soldermask layers included so it shows a realistic board, not a white slab.

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
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from kicraft.design.synthesis.models3d import stage_3d_models  # noqa: E402

SAMPLES_DIR = REPO / "kicraft" / "server" / "sample_projects"
_MODEL_RE = re.compile(r'\(model "([^"]+)"')

# Roots searched (in order) to recover a model file by basename when the board
# emits a bare ``/X.wrl`` ref that resolves nowhere: the home parts cache first
# (freshest, the exact fetched models) then the vendored parts library (generic
# package bodies — SOT-23, etc. — share a basename across parts). Each holds
# ``<lib>/3d/<file>``.
_MODEL_ROOTS = (Path.home() / ".kicraft" / "parts",
                REPO / "kicraft" / "parts_library")

# Flags that turn the GLB from a bare white substrate into a realistic board:
# the conductor / silkscreen / soldermask layers (copper tracks, pads, zones,
# white silk, coloured mask). ``--subst-models`` is required, not optional — the
# GLB exporter is OpenCASCADE-based and cannot load VRML, so it substitutes each
# footprint's STEP sibling for its .wrl to bring the component bodies in (without
# it kicad-cli aborts with "Cannot load any VRML model for this export").
_GLB_LAYER_FLAGS = ("--subst-models", "--include-tracks", "--include-pads",
                    "--include-zones", "--include-silkscreen",
                    "--include-soldermask", "--cut-vias-in-body")


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


def _find_model_source(basename: str) -> Path | None:
    """First on-disk model file matching ``basename`` across the recovery roots."""
    for root in _MODEL_ROOTS:
        if not root.is_dir():
            continue
        hit = next((c for c in sorted(root.glob(f"*/3d/{basename}")) if c.is_file()),
                   None)
        if hit is not None:
            return hit
    return None


def _resolve_model_refs(dest: Path, stem: str) -> tuple[int, list[str]]:
    """Make every component 3D-model ref in ``dest/<stem>.kicad_pcb`` resolve.

    Stock ``${KICAD9_3DMODEL_DIR}`` refs resolve through system KiCad and refs
    already pointing at ``${KIPRJMOD}/3dmodels`` are left alone. Every other ref
    (routed boards emit bundle models as a bare ``/X.wrl`` that resolves nowhere)
    is matched by basename: first against models already staged under
    ``dest/3dmodels``, then against the parts cache / vendored library, copying
    the model (and its STEP sibling) into ``dest/3dmodels/<lib>/`` so the GLB
    export and a downloaded zip both resolve it. Returns
    ``(refs_rewritten, unresolved_paths)``.
    """
    staged = {f.name: f.relative_to(dest).as_posix()
              for f in sorted(dest.glob("3dmodels/*/*")) if f.is_file()}
    unresolved: list[str] = []
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        path = m.group(1)
        if path.startswith("${KICAD9_3DMODEL_DIR}"):
            return m.group(0)  # stock model, resolves through system KiCad
        base = path.rsplit("/", 1)[-1]
        target = staged.get(base)
        if target is None:
            src = _find_model_source(base)
            if src is not None:
                lib_dir = dest / "3dmodels" / src.parent.parent.name
                lib_dir.mkdir(parents=True, exist_ok=True)
                for sib in (src, src.with_suffix(".step"), src.with_suffix(".stp")):
                    if sib.is_file():
                        shutil.copy2(sib, lib_dir / sib.name)
                target = (lib_dir / src.name).relative_to(dest).as_posix()
                staged[base] = target
        if target is None:
            if not path.startswith("${KIPRJMOD}/3dmodels/"):
                unresolved.append(path)
            return m.group(0)
        want = "${KIPRJMOD}/" + target
        if path == want:
            return m.group(0)
        n += 1
        return f'(model "{want}"'

    pcb = dest / f"{stem}.kicad_pcb"
    pcb.write_text(_MODEL_RE.sub(repl, pcb.read_text()))
    return n, unresolved


def _stage_and_fix_models(proj: Path, dest: Path, stem: str,
                          bom_parts: list[dict]) -> tuple[int, list[str]]:
    """Stage bundle models into dest/3dmodels and make every board ref resolve."""
    bom = SimpleNamespace(parts=[SimpleNamespace(footprint=p.get("footprint", ""))
                                 for p in bom_parts])
    stage_3d_models(dest, bom, project_root=REPO)
    return _resolve_model_refs(dest, stem)


def render_board_png(dest: Path, stem: str, *, out: Path, rotate: str, zoom: str,
                     width: int, height: int, background: str = "transparent",
                     floor: bool = False, perspective: bool = False) -> None:
    """Raytrace the assembled board to ``out`` (poster / og:image / hero image)."""
    extra: list[str] = []
    if floor:
        extra.append("--floor")
    if perspective:
        extra.append("--perspective")
    _run(["kicad-cli", "pcb", "render", "-o", str(out), "--quality", "high",
          "--background", background, "--rotate", rotate, "--zoom", zoom,
          "-w", str(width), "-h", str(height), *extra,
          "-D", f"KIPRJMOD={dest}", str(dest / f"{stem}.kicad_pcb")])


# kicad-cli's GLB export emits component geometry with NO material (the STEP/VRML
# colours are dropped), so every part renders default white in model-viewer. We
# repaint each component from its own model afterwards. Stock passives ship only a
# STEP (no .wrl to read), so fall back to a per-reference body colour.
_KICAD9_3DMODEL_DIR = Path("/usr/share/kicad/3dmodels")
_REF_BODY_COLOR = {  # reference-prefix -> plausible body colour (linear-ish RGB)
    "R": (0.10, 0.10, 0.11),   # chip resistor (dark body)
    "C": (0.78, 0.66, 0.46),   # MLCC (tan ceramic)
    "L": (0.16, 0.16, 0.16), "FB": (0.16, 0.16, 0.16),
    "D": (0.13, 0.13, 0.14), "LED": (0.85, 0.85, 0.85),
    "U": (0.09, 0.09, 0.10), "Q": (0.09, 0.09, 0.10),
    "J": (0.62, 0.62, 0.64), "SW": (0.13, 0.13, 0.14),
    "Y": (0.70, 0.70, 0.72),
}


def _dominant_wrl_color(wrl: Path) -> tuple | None:
    """The most common ``diffuseColor`` in a VRML model ~ its body colour."""
    cols = re.findall(r"diffuseColor\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
                      wrl.read_text(errors="ignore"))
    if not cols:
        return None
    counts = Counter(tuple(round(float(x), 4) for x in t) for t in cols)
    return counts.most_common(1)[0][0]


def _component_color(ref: str, model_path: str | None, dest: Path) -> tuple:
    """Body colour for ``ref``: the dominant colour of its bundled .wrl when one
    resolves (the same colours the raytraced render uses), else a per-reference
    fallback so stock-STEP passives still read right."""
    base = None
    if model_path and model_path.startswith("${KIPRJMOD}/"):
        base = dest / model_path[len("${KIPRJMOD}/"):]
    elif model_path and model_path.startswith("${KICAD9_3DMODEL_DIR}/"):
        base = _KICAD9_3DMODEL_DIR / model_path[len("${KICAD9_3DMODEL_DIR}/"):]
    if base is not None:
        wrl = base.with_suffix(".wrl")
        if wrl.is_file():
            col = _dominant_wrl_color(wrl)
            if col:
                return col
    prefix = (re.match(r"[A-Za-z]+", ref or "") or [""])[0].upper()
    for key in (prefix, prefix[:2], prefix[:1]):
        if key in _REF_BODY_COLOR:
            return _REF_BODY_COLOR[key]
    return (0.40, 0.40, 0.42)


def _board_ref_colors(board_pcb: Path) -> dict[str, tuple]:
    """``reference -> body colour`` for every footprint on the board."""
    dest = board_pcb.parent
    out: dict[str, tuple] = {}
    for block in re.split(r"(?=\(footprint )", board_pcb.read_text())[1:]:
        ref = re.search(r'\(property "Reference" "([^"]+)"', block)
        if not ref:
            continue
        model = re.search(r'\(model "([^"]+)"', block)
        out[ref.group(1)] = _component_color(ref.group(1), model and model.group(1),
                                              dest)
    return out


def recolor_glb_components(glb: Path, board_pcb: Path) -> int:
    """Give each component mesh in ``glb`` a material in its real body colour.

    kicad-cli leaves component primitives material-less (→ white). We append one
    PBR material per component node (named by reference designator) and point all
    its primitives at it. Only the JSON chunk changes; the binary geometry buffer
    is untouched. Returns the number of components recoloured.
    """
    ref_colors = _board_ref_colors(board_pcb)
    data = bytearray(glb.read_bytes())
    magic, ver, _total = struct.unpack_from("<III", data, 0)
    off = 12
    json_len, json_type = struct.unpack_from("<II", data, off)
    off += 8
    j = json.loads(bytes(data[off:off + json_len]))
    rest = bytes(data[off + json_len:])  # BIN chunk(s), kept verbatim

    meshes = j.get("meshes", [])
    materials = j.setdefault("materials", [])
    n = 0
    for node in j.get("nodes", []):
        ref, mi = node.get("name"), node.get("mesh")
        if mi is None or ref not in ref_colors:
            continue
        r, g, b = ref_colors[ref]
        metal = min(r, g, b) > 0.55 and (max(r, g, b) - min(r, g, b)) < 0.05
        idx = len(materials)
        materials.append({
            "name": f"cmp_{ref}",
            "pbrMetallicRoughness": {
                "baseColorFactor": [r, g, b, 1.0],
                "metallicFactor": 0.9 if metal else 0.0,
                "roughnessFactor": 0.35 if metal else 0.55,
            },
        })
        for prim in meshes[mi].get("primitives", []):
            prim["material"] = idx
        n += 1

    new_json = json.dumps(j, separators=(",", ":")).encode("utf-8")
    new_json += b" " * ((-len(new_json)) % 4)  # 4-byte align, pad with spaces
    out = bytearray(struct.pack("<III", magic, ver, 0))
    out += struct.pack("<II", len(new_json), json_type)
    out += new_json
    out += rest
    struct.pack_into("<I", out, 8, len(out))  # total file length
    glb.write_bytes(out)
    return n


def export_board_glb(dest: Path, stem: str) -> int:
    """Export ``previews/board.glb`` with board layers + bodies, then repaint each
    component in its real colour. Returns the number of components recoloured."""
    glb = dest / "previews" / "board.glb"
    _run(["kicad-cli", "pcb", "export", "glb", "--force", *_GLB_LAYER_FLAGS,
          "-D", f"KIPRJMOD={dest}", "-o", str(glb),
          str(dest / f"{stem}.kicad_pcb")])
    return recolor_glb_components(glb, dest / f"{stem}.kicad_pcb")


def promote(run_dir: Path, sample_id: str, *, rotate: str, zoom: str,
            width: int, height: int, background: str = "transparent",
            featured: bool = False) -> dict:
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

    n_fixed, unresolved = _stage_and_fix_models(proj, dest, stem, bom_parts)
    if unresolved:
        print(f"  WARNING: {len(unresolved)} model ref(s) still unresolved (no "
              f"source found): {', '.join(sorted(set(unresolved)))}", file=sys.stderr)

    # The landing page is all stills: a flat, mostly-top-down isometric grounded
    # with a floor shadow (transparent bg → the board floats on the page). The
    # featured board also gets a larger hero.png.
    render_board_png(dest, stem, out=dest / "previews" / "board.png", rotate=rotate,
                     zoom=zoom, width=width, height=height, background=background,
                     floor=True, perspective=True)
    if featured:
        render_board_png(dest, stem, out=dest / "previews" / "hero.png",
                         rotate=rotate, zoom=zoom, width=2200, height=1650,
                         background=background, floor=True, perspective=True)
    n_recolored = export_board_glb(dest, stem)

    sheets = len(list(dest.glob("*.kicad_sch")))
    parts = len(bom_parts) or \
        (dest / f"{stem}.kicad_pcb").read_text().count("(footprint ")
    return {"dest": dest, "stem": stem, "sheets": sheets, "parts": parts,
            "models_fixed": n_fixed, "unresolved": unresolved,
            "recolored": n_recolored,
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
    ap.add_argument("--rotate", default="-24,0,16",
                    help="kicad-cli render --rotate (flat, mostly top-down iso)")
    ap.add_argument("--zoom", default="0.86", help="kicad-cli render --zoom")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--background", default="transparent",
                    help="render bg: transparent (default), opaque, default")
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    info = promote(run_dir, args.id, rotate=args.rotate, zoom=args.zoom,
                   width=args.width, height=args.height, background=args.background,
                   featured=args.featured)

    prompt = args.prompt
    if not prompt:
        bt = run_dir / "brief.txt"
        prompt = bt.read_text().strip() if bt.is_file() else ""

    print(f"\npromoted -> {info['dest']}")
    print(f"  stem={info['stem']} sheets={info['sheets']} parts={info['parts']} "
          f"models_rewritten={info['models_fixed']} recolored={info['recolored']} "
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
