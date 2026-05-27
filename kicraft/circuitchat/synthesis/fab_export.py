"""Fab-output export: Gerbers + drill + placement (CPL) + BOM, zipped.

Wraps ``kicad-cli pcb export {gerbers,drill,pos}`` to turn a routed ``.kicad_pcb``
into a JLCPCB/OSHPark-ready package:

* Gerber X2 for the standard fab layer stack (copper, mask, silk, paste, edge),
* Excellon drill files (+ a drill map, PTH/NPTH separated),
* a CSV placement / CPL file, and
* a BOM CSV derived from the BOM slot,

all collected under ``<out_dir>/fab/`` and zipped to
``<out_dir>/<stem>_fab_<UTCdate>.zip``.
"""
from __future__ import annotations

import csv
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_KICAD_CLI = "kicad-cli"
# Standard 2-layer fab stack. KiCad-9 untranslated layer names.
_FAB_LAYERS = (
    "F.Cu,B.Cu,F.Paste,B.Paste,F.Silkscreen,B.Silkscreen,F.Mask,B.Mask,Edge.Cuts"
)
_LCSC_RE = re.compile(r"\bC\d{4,}\b")


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd[:5])} ... failed (rc={r.returncode}): "
            f"{(r.stderr or r.stdout).strip()[:400]}"
        )


def _write_bom_csv(path: Path, parts: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ref", "value", "footprint", "mpn", "lcsc", "sheet"])
        for p in parts:
            note = p.get("sourcing_note") or ""
            m = _LCSC_RE.search(note)
            w.writerow([
                p.get("ref", ""),
                p.get("value", ""),
                p.get("footprint", ""),
                p.get("mpn") or "",
                m.group(0) if m else "",
                p.get("sheet", ""),
            ])


def export_fab(
    pcb_path: str,
    out_dir: str,
    stem: str,
    *,
    bom_parts: list[dict[str, Any]] | None = None,
    fab_layers: str = _FAB_LAYERS,
) -> dict[str, Any]:
    """Export Gerbers/drill/CPL/BOM from a routed PCB and zip them.

    Returns {fab_dir, zip, files, bom_csv}. Raises RuntimeError if any
    kicad-cli step fails.
    """
    pcb_path = str(pcb_path)
    out = Path(out_dir)
    fab = out / "fab"
    fab.mkdir(parents=True, exist_ok=True)

    # Gerbers (X2) for the standard fab stack.
    _run([
        _KICAD_CLI, "pcb", "export", "gerbers",
        "-o", str(fab) + "/", "-l", fab_layers, pcb_path,
    ])
    # Excellon drill + map, PTH/NPTH separated.
    _run([
        _KICAD_CLI, "pcb", "export", "drill",
        "-o", str(fab) + "/", "--format", "excellon",
        "--excellon-separate-th", "--generate-map", pcb_path,
    ])
    # Placement / CPL (both sides, mm, CSV).
    _run([
        _KICAD_CLI, "pcb", "export", "pos",
        "-o", str(fab / f"{stem}-pos.csv"), "--format", "csv", "--units", "mm",
        pcb_path,
    ])

    bom_csv: Path | None = None
    if bom_parts:
        bom_csv = fab / "bom.csv"
        _write_bom_csv(bom_csv, bom_parts)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    zip_path = out / f"{stem}_fab_{ts}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(fab.iterdir()):
            if f.is_file():
                zf.write(f, f.name)

    files = [f.name for f in sorted(fab.iterdir()) if f.is_file()]
    return {
        "fab_dir": str(fab),
        "zip": str(zip_path),
        "files": files,
        "bom_csv": str(bom_csv) if bom_csv else None,
    }
