"""Fab-output export: Gerbers + drill + placement (CPL) + BOM, zipped.

Wraps ``kicad-cli pcb export {gerbers,drill,pos}`` to turn a routed ``.kicad_pcb``
into a JLCPCB/OSHPark-ready package:

* Gerber X2 for the standard fab layer stack (copper, mask, silk, paste, edge),
* Excellon drill files (+ a drill map, PTH/NPTH separated),
* a CSV placement / CPL file,
* a BOM CSV derived from the BOM slot, and
* best-effort 3D outputs: a STEP model and a rendered PNG of the assembled
  board (``kicad-cli pcb export step`` / ``pcb render``; never fail the build),

all collected under ``<out_dir>/fab/`` and zipped to
``<out_dir>/<stem>_fab_<UTCdate>.zip``.
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_KICAD_CLI = "kicad-cli"
# Standard 2-layer fab stack. KiCad-9 untranslated layer names.
_FAB_LAYERS = "F.Cu,B.Cu,F.Paste,B.Paste,F.Silkscreen,B.Silkscreen,F.Mask,B.Mask,Edge.Cuts"
_LCSC_RE = re.compile(r"\bC\d{4,}\b")
# Chip R/C/L imperial size codes that LOOK like C-numbers when prefixed with
# 'C' in prose ("100nF X7R, package C0603") -- never LCSC pins. Real LCSC
# part numbers also never lead with 0, which excludes C0201/C0402/... and
# C01005 by shape alone; this set catches the nonzero-led sizes too.
_PACKAGE_SIZE_CODES = frozenset(
    {
        "1008",
        "1111",
        "1206",
        "1210",
        "1218",
        "1812",
        "1825",
        "2010",
        "2220",
        "2225",
        "2512",
        "2920",
    }
)


def extract_lcsc_pin(text: str) -> str | None:
    """The explicit LCSC C-number pinned in prose, or None.

    Shared by the BOM sourcing resolution (cli_app) and the fab BOM export so
    both always agree on what counts as a pin. Package-size tokens (C0603,
    C1206, ...) are excluded: treating "package C0603" as a pin either bounced
    correct BOM commits at the sourcing gate or exported an unrelated real
    part into the fab BOM.
    """
    for m in _LCSC_RE.finditer(text or ""):
        digits = m.group(0)[1:]
        if digits.startswith("0"):
            continue
        if digits in _PACKAGE_SIZE_CODES:
            # Ambiguous token: C1812 is BOTH a chip size code and a real,
            # in-stock LCSC part (3.6pF C0G 0805). An explicit "LCSC" right
            # before the token disambiguates in favor of a pin
            # ("LCSC C1812"); bare prose ("package C1812") stays excluded
            # (2026-07-19 review §4.7).
            prefix = (text or "")[max(0, m.start() - 16) : m.start()].lower()
            if "lcsc" not in prefix:
                continue
        return m.group(0)
    return None


def _run(cmd: list[str], timeout: float | None = None) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
            if p.get("assembly", True) is False:
                continue
            note = p.get("sourcing_note") or ""
            w.writerow(
                [
                    p.get("ref", ""),
                    p.get("value", ""),
                    p.get("footprint", ""),
                    p.get("mpn") or "",
                    extract_lcsc_pin(note) or "",
                    p.get("sheet", ""),
                ]
            )


def export_fab(
    pcb_path: str,
    out_dir: str,
    stem: str,
    *,
    bom_parts: list[dict[str, Any]] | None = None,
    fab_layers: str = _FAB_LAYERS,
    include_3d: bool = True,
) -> dict[str, Any]:
    """Export Gerbers/drill/CPL/BOM from a routed PCB and zip them.

    With ``include_3d`` (the default) also exports a STEP model
    (``<stem>.step``, models substituted from the footprints' WRL refs) and
    a rendered PNG of the assembled board (``board_3d.png``); both land in
    the zip. The 3D outputs are best-effort: the gerbers are the
    deliverable, so a STEP/render failure warns and continues.

    Returns {fab_dir, zip, files, bom_csv, step, board_3d_png}. Raises
    RuntimeError if any non-3D kicad-cli step fails.
    """
    pcb_path = str(pcb_path)
    out = Path(out_dir)
    fab = out / "fab"
    fab.mkdir(parents=True, exist_ok=True)

    # Gerbers (X2) for the standard fab stack.
    _run(
        [
            _KICAD_CLI,
            "pcb",
            "export",
            "gerbers",
            "-o",
            str(fab) + "/",
            "-l",
            fab_layers,
            pcb_path,
        ]
    )
    # Excellon drill + map, PTH/NPTH separated.
    _run(
        [
            _KICAD_CLI,
            "pcb",
            "export",
            "drill",
            "-o",
            str(fab) + "/",
            "--format",
            "excellon",
            "--excellon-separate-th",
            "--generate-map",
            pcb_path,
        ]
    )
    # Placement / CPL (both sides, mm, CSV).
    _run(
        [
            _KICAD_CLI,
            "pcb",
            "export",
            "pos",
            "-o",
            str(fab / f"{stem}-pos.csv"),
            "--format",
            "csv",
            "--units",
            "mm",
            pcb_path,
        ]
    )

    bom_csv: Path | None = None
    if bom_parts:
        bom_csv = fab / "bom.csv"
        _write_bom_csv(bom_csv, bom_parts)

    step_path: Path | None = None
    render_path: Path | None = None
    if include_3d:
        step_candidate = fab / f"{stem}.step"
        try:
            _run(
                [
                    _KICAD_CLI,
                    "pcb",
                    "export",
                    "step",
                    "--subst-models",
                    "--force",
                    "-o",
                    str(step_candidate),
                    pcb_path,
                ],
                timeout=300,
            )
            step_path = step_candidate
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"fab: STEP export failed, continuing without: {exc}", file=sys.stderr)

        render_candidate = fab / "board_3d.png"
        # Never let a failed re-render resurrect a stale image into the zip.
        render_candidate.unlink(missing_ok=True)
        render_cmd = [
            _KICAD_CLI,
            "pcb",
            "render",
            "-o",
            str(render_candidate),
            "--quality",
            "high",
            "--background",
            "opaque",
            "--rotate",
            "-30,0,30",
            "--zoom",
            "0.9",
            "-w",
            "1600",
            "-h",
            "1200",
            pcb_path,
        ]
        try:
            try:
                _run(render_cmd, timeout=300)
            except (RuntimeError, subprocess.TimeoutExpired):
                # Headless boxes have no GL context; xvfb-run provides one.
                if not shutil.which("xvfb-run"):
                    raise
                _run(["xvfb-run", "-a", *render_cmd], timeout=300)
            render_path = render_candidate
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"fab: 3D render failed, continuing without: {exc}", file=sys.stderr)

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
        "step": str(step_path) if step_path else None,
        "board_3d_png": str(render_path) if render_path else None,
    }
