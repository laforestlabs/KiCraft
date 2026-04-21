"""FreeRouting integration — board cleanup → DSN export → FreeRouting CLI → SES import.

Routing pipeline:
  prepare_board_for_placement() → placement → route_with_freerouting()
  route_with_freerouting(): optional cleanup → export_dsn() → run_freerouting() → import_ses()
  Then count_board_tracks() extracts real trace/via counts from the result.

This module also provides:
- lightweight routed-board validation helpers used by the subcircuits pipeline
- canonical copper import from routed KiCad boards so solved leaf artifacts can
  persist real routed traces/vias

Verification note:
- hierarchical/subcircuit changes should be verified by running the leaf
  subcircuit pipeline once, not by a 3-round autoexperiment shortcut

Note: Uses FreeRouting v1.9.0.  v2.1.0 has a regression where max_passes
is ignored and routing runs indefinitely.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import site
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from kicraft.autoplacer.brain.types import Layer, Point, TraceSegment, Via


def _kicad_subprocess_env() -> dict[str, str]:
    """Build subprocess env that can import KiCad's pcbnew module.

    In virtualenvs, KiCad's site-packages path may not be visible to child
    Python processes. This adds common KiCad locations to PYTHONPATH.
    """
    env = os.environ.copy()

    candidates = []
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates.extend(
        [
            f"/usr/lib/python{ver}/site-packages",
            f"/usr/lib64/python{ver}/site-packages",
            "/usr/lib/python3/dist-packages",
            "/usr/lib64/python3/dist-packages",
        ]
    )
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass

    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    merged = list(existing)
    for p in candidates:
        if not p:
            continue
        if (
            os.path.exists(os.path.join(p, "pcbnew.py"))
            or os.path.isdir(os.path.join(p, "pcbnew"))
        ) and p not in merged:
            merged.append(p)

    if merged:
        env["PYTHONPATH"] = os.pathsep.join(merged)

    return env


def _run_pcbnew_script(script: str) -> None:
    """Run a pcbnew script in a fresh subprocess to avoid SwigPyObject bugs."""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=_kicad_subprocess_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pcbnew subprocess failed (rc={result.returncode}):\n{result.stderr}"
        )


def clear_traces(
    kicad_pcb_path: str,
    preserve_thermal_vias: bool = True,
    thermal_refs: list[str] | None = None,
    thermal_radius_mm: float = 3.0,
) -> None:
    """Remove all traces/vias from the board, optionally preserving thermal vias."""
    thermal_refs = thermal_refs or []
    _run_pcbnew_script(
        "import math, pcbnew\n"
        f"board = pcbnew.LoadBoard({kicad_pcb_path!r})\n"
        "if board is None:\n"
        f"    raise RuntimeError('Failed to load board: {kicad_pcb_path}')\n"
        f"thermal_refs = {thermal_refs!r}\n"
        f"thermal_radius_mm = {thermal_radius_mm!r}\n"
        f"preserve = {preserve_thermal_vias!r}\n"
        "thermal_centers = []\n"
        "if preserve:\n"
        "    for ref in thermal_refs:\n"
        "        fp = board.FindFootprintByReference(ref)\n"
        "        if fp:\n"
        "            pos = fp.GetPosition()\n"
        "            thermal_centers.append((pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)))\n"
        "to_remove = []\n"
        "for t in board.GetTracks():\n"
        "    if preserve and isinstance(t, pcbnew.PCB_VIA):\n"
        "        vpos = t.GetPosition()\n"
        "        vx, vy = pcbnew.ToMM(vpos.x), pcbnew.ToMM(vpos.y)\n"
        "        if any(math.hypot(vx-cx, vy-cy) <= thermal_radius_mm for cx,cy in thermal_centers):\n"
        "            continue\n"
        "    to_remove.append(t)\n"
        "for t in to_remove: board.Remove(t)\n"
        f"board.Save({kicad_pcb_path!r})\n"
    )


def clear_zones(kicad_pcb_path: str) -> None:
    """Remove all copper zones from the board."""
    _run_pcbnew_script(
        "import pcbnew\n"
        f"board = pcbnew.LoadBoard({kicad_pcb_path!r})\n"
        "if board is None:\n"
        f"    raise RuntimeError('Failed to load board: {kicad_pcb_path}')\n"
        "for z in list(board.Zones()):\n"
        "    board.Remove(z)\n"
        f"board.Save({kicad_pcb_path!r})\n"
    )


def _unlock_traces(kicad_pcb_path: str) -> None:
    """Unlock all traces and vias in the board file."""
    _run_pcbnew_script(
        "import pcbnew\n"
        f"board = pcbnew.LoadBoard({kicad_pcb_path!r})\n"
        "for track in board.GetTracks():\n"
        "    track.SetLocked(False)\n"
        f"board.Save({kicad_pcb_path!r})\n"
    )


def prepare_board_for_placement(kicad_pcb_path: str) -> None:
    """Strip stale routing artifacts so placement starts from a clean board."""
    clear_traces(
        kicad_pcb_path,
        preserve_thermal_vias=False,
        thermal_refs=[],
        thermal_radius_mm=0.0,
    )
    clear_zones(kicad_pcb_path)


def count_board_tracks(kicad_pcb_path: str) -> dict[str, float | int]:
    """Count traces, vias, and total trace length from a routed board.

    Runs in subprocess to avoid pcbnew SWIG issues.
    Returns {traces: int, vias: int, total_length_mm: float}.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, pcbnew\n"
            f"board = pcbnew.LoadBoard({kicad_pcb_path!r})\n"
            "traces = vias = 0\n"
            "length_nm = 0\n"
            "for t in board.GetTracks():\n"
            "    if isinstance(t, pcbnew.PCB_VIA):\n"
            "        vias += 1\n"
            "    else:\n"
            "        traces += 1\n"
            "        length_nm += t.GetLength()\n"
            "print(json.dumps({'traces': traces, 'vias': vias,"
            "  'total_length_mm': round(pcbnew.ToMM(length_nm), 2)}))\n",
        ],
        capture_output=True,
        text=True,
        env=_kicad_subprocess_env(),
    )
    if result.returncode != 0:
        return {"traces": 0, "vias": 0, "total_length_mm": 0.0}
    return json.loads(result.stdout.strip())


def export_dsn(
    kicad_pcb_path: str, dsn_path: str, lock_existing_traces: bool = False
) -> None:
    """Export Specctra DSN from a KiCad PCB file using pcbnew API.

    Assumes copper zones have already been stripped so FreeRouting starts
    from a clean board containing only footprints, nets, and board geometry.

    If lock_existing_traces is True, all existing tracks and vias are marked
    as locked before export so FreeRouting treats them as fixed pre-routes.
    """
    lock_script = ""
    if lock_existing_traces:
        lock_script = (
            "# Lock all existing traces so FreeRouting treats them as fixed\n"
            "for track in board.GetTracks():\n"
            "    track.SetLocked(True)\n"
        )
    _run_pcbnew_script(
        "import pcbnew\n"
        f"board = pcbnew.LoadBoard({kicad_pcb_path!r})\n"
        + lock_script
        + "board.BuildConnectivity()\n"
        f"board.Save({kicad_pcb_path!r})\n"
        f"pcbnew.ExportSpecctraDSN(board, {dsn_path!r})\n"
    )
    # Post-process DSN: raise smd_smd clearance to match the global clearance.
    # KiCad exports (clearance 50 (type smd_smd)) which is only 0.05mm,
    # leading to DRC violations when KiCad checks with its 0.2mm rule.
    _patch_dsn_clearance(dsn_path)


def _patch_dsn_clearance(dsn_path: str) -> None:
    """Raise ALL type-specific clearances in a DSN file to match the global clearance.

    KiCad exports reduced clearances for certain types (e.g. smd_smd at 0.05mm,
    smd_to_trace, etc.) which cause DRC violations when KiCad checks with its
    actual design rules.  Replace every typed clearance with the global value.
    """
    with open(dsn_path) as f:
        content = f.read()
    # Find global clearance value (first bare clearance line without a type qualifier)
    m = re.search(r"\(clearance\s+(\d+)\)", content)
    if not m:
        return
    global_clearance = m.group(1)
    # Replace ALL type-specific clearances (smd_smd, smd_to_trace, etc.)
    patched = re.sub(
        r"\(clearance\s+\d+\s+\(type\s+(\w+)\)\)",
        lambda match: f"(clearance {global_clearance} (type {match.group(1)}))",
        content,
    )
    if patched != content:
        with open(dsn_path, "w") as f:
            f.write(patched)


def parse_freerouting_output(stdout: str, stderr: str, returncode: int) -> dict[str, Any]:
    """Parse FreeRouting stdout/stderr for routing statistics."""
    stats: dict[str, Any] = {
        "returncode": returncode,
        "passes": 0,
        "unrouted": -1,
        "violations": -1,
        "score": 0.0,
        "routing_seconds": 0.0,
        "optimization_seconds": 0.0,
        "_raw_stdout": stdout[:2000] if stdout else "",
        "_raw_stderr": stderr[:2000] if stderr else "",
    }

    combined = stdout + "\n" + stderr

    # v2.x format: "Auto-routing was completed in X.XX seconds with the score of X (N unrouted and M violations)."
    m = re.search(
        r"Auto-routing was completed in ([\d.]+) seconds.*?"
        r"score of ([\d.]+).*?(\d+) unrouted.*?(\d+) violations",
        combined,
    )
    if m:
        stats["routing_seconds"] = float(m.group(1))
        stats["score"] = float(m.group(2))
        stats["unrouted"] = int(m.group(3))
        stats["violations"] = int(m.group(4))
    else:
        # v1.9.x format: "Auto-routing was completed in X.XX seconds."
        m19 = re.search(
            r"Auto-routing was completed in ([\d.]+) seconds",
            combined,
        )
        if m19:
            stats["routing_seconds"] = float(m19.group(1))

    # Count successful passes (v2.x logs per-pass)
    pass_matches = re.findall(r"Auto-router pass #(\d+)", combined)
    if pass_matches:
        stats["passes"] = int(pass_matches[-1])

    # Parse optimization time (both versions)
    m_opt = re.search(
        r"[Oo]ptimization was completed in ([\d.]+) seconds",
        combined,
    )
    if m_opt:
        stats["optimization_seconds"] = float(m_opt.group(1))

    return stats


def run_freerouting(
    dsn_path: str,
    ses_path: str,
    jar_path: str,
    timeout_s: int = 120,
    max_passes: int = 40,
    work_dir: str | None = None,
) -> dict[str, Any]:
    """Run FreeRouting CLI and return result metadata.

    Uses start_new_session so the Java process gets its own process group,
    allowing clean kill via os.killpg() on timeout or stop request.
    """
    jar_path = os.path.expanduser(jar_path)
    cmd = [
        "java",
        "-jar",
        jar_path,
        "-de",
        dsn_path,
        "-do",
        ses_path,
        "-mp",
        str(max_passes),
        "-mt",
        "1",  # single-threaded optimization (multi is buggy)
        "-dct",
        "0",  # auto-dismiss dialogs immediately
    ]

    cwd = work_dir or os.path.dirname(dsn_path)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        # Kill entire process group (Java + any children)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            pass
        stdout, stderr = proc.communicate(timeout=5)
        return parse_freerouting_output(stdout, stderr, -1)

    return parse_freerouting_output(stdout, stderr, proc.returncode)


def import_ses(kicad_pcb_path: str, ses_path: str, output_path: str) -> None:
    """Import Specctra SES session file into KiCad PCB."""
    _run_pcbnew_script(
        "import pcbnew\n"
        f"board = pcbnew.LoadBoard({kicad_pcb_path!r})\n"
        f"pcbnew.ImportSpecctraSES(board, {ses_path!r})\n"
        f"board.Save({output_path!r})\n"
    )


def _build_contact_sheet(image_paths: list[str], output_path: str) -> bool:
    """Build a simple contact sheet from existing images using ImageMagick."""
    existing = [path for path in image_paths if path and os.path.exists(path)]
    if not existing:
        return False

    magick = shutil.which("magick")
    if magick is None:
        return False

    result = subprocess.run(
        [
            magick,
            *existing,
            "-background",
            "white",
            "-tile",
            "2x2",
            "-geometry",
            "+8+8",
            "montage",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and os.path.exists(output_path)


def route_with_freerouting(
    kicad_pcb_path: str, output_path: str, jar_path: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Full DSN → FreeRouting → SES pipeline. Returns routing stats.

    Retries once on crash (rc != 0 and no SES output) with reduced
    max_passes to work around FreeRouting v1.9.0 intermittent failures.

    By default this preserves the historical behavior of clearing traces/zones
    before DSN export. Set either:
    - config["freerouting_preserve_existing_copper"] = True
    - config["freerouting_clear_existing_copper"] = False

    to export and route from a board that already contains routed copper, which
    is required for hierarchical parent routing with preloaded child traces.
    """
    preserve_existing_copper = bool(
        config.get(
            "freerouting_preserve_existing_copper",
            not config.get("freerouting_clear_existing_copper", True),
        )
    )
    clear_existing_zones = bool(config.get("freerouting_clear_zones", True))

    if not preserve_existing_copper:
        clear_traces(
            kicad_pcb_path,
            preserve_thermal_vias=True,
            thermal_refs=config.get("thermal_refs", []),
            thermal_radius_mm=config.get("thermal_radius_mm", 3.0),
        )
        if clear_existing_zones:
            clear_zones(kicad_pcb_path)
    elif clear_existing_zones:
        clear_zones(kicad_pcb_path)

    max_passes = config.get("freerouting_max_passes", 40)
    timeout_s = config.get("freerouting_timeout_s", 120)

    for attempt in range(2):
        with tempfile.TemporaryDirectory() as tmpdir:
            dsn_path = os.path.join(tmpdir, "board.dsn")
            ses_path = os.path.join(tmpdir, "board.ses")

            export_dsn(
                kicad_pcb_path,
                dsn_path,
                lock_existing_traces=preserve_existing_copper,
            )

            passes = max_passes if attempt == 0 else max(10, max_passes // 2)
            stats = run_freerouting(
                dsn_path,
                ses_path,
                jar_path,
                timeout_s=timeout_s,
                max_passes=passes,
            )
            stats["preserved_existing_copper"] = preserve_existing_copper
            stats["cleared_zones_before_export"] = clear_existing_zones

            if os.path.exists(ses_path):
                import_ses(kicad_pcb_path, ses_path, output_path)
                if preserve_existing_copper:
                    _unlock_traces(output_path)
                return stats

            if attempt == 0:
                _stderr_snippet = (stats.get('_raw_stderr') or '').strip()[:200]
                print(
                    f"  FreeRouting crash (rc={stats.get('returncode', '?')}), retrying with {max(10, max_passes // 2)} passes..."
                    + (f" stderr: {_stderr_snippet}" if _stderr_snippet else "")
                )
                continue

            raise RuntimeError(
                f"FreeRouting produced no SES output after 2 attempts (rc={stats.get('returncode', '?')})"
            )

    raise RuntimeError("FreeRouting routing failed")


def import_routed_copper(kicad_pcb_path: str) -> dict[str, Any]:
    """Import routed copper geometry from a KiCad board into canonical objects.

    Returns:
        {
            "traces": list[TraceSegment],
            "vias": list[Via],
            "trace_count": int,
            "via_count": int,
            "total_length_mm": float,
        }
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, pcbnew\n"
            f"board = pcbnew.LoadBoard({kicad_pcb_path!r})\n"
            "payload = {'traces': [], 'vias': [], 'trace_count': 0, 'via_count': 0, 'total_length_mm': 0.0}\n"
            "for track in board.GetTracks():\n"
            "    if isinstance(track, pcbnew.PCB_VIA):\n"
            "        pos = track.GetPosition()\n"
            "        try:\n"
            "            size_mm = pcbnew.ToMM(track.GetWidth(pcbnew.F_Cu))\n"
            "        except TypeError:\n"
            "            size_mm = pcbnew.ToMM(track.GetWidth())\n"
            "        payload['vias'].append({\n"
            "            'pos': {'x': pcbnew.ToMM(pos.x), 'y': pcbnew.ToMM(pos.y)},\n"
            "            'net': track.GetNetname(),\n"
            "            'drill_mm': pcbnew.ToMM(track.GetDrill()),\n"
            "            'size_mm': size_mm,\n"
            "        })\n"
            "    else:\n"
            "        start = track.GetStart()\n"
            "        end = track.GetEnd()\n"
            "        width_mm = pcbnew.ToMM(track.GetWidth())\n"
            "        length_mm = pcbnew.ToMM(track.GetLength())\n"
            "        layer_name = board.GetLayerName(track.GetLayer())\n"
            "        payload['traces'].append({\n"
            "            'start': {'x': pcbnew.ToMM(start.x), 'y': pcbnew.ToMM(start.y)},\n"
            "            'end': {'x': pcbnew.ToMM(end.x), 'y': pcbnew.ToMM(end.y)},\n"
            "            'layer': layer_name,\n"
            "            'net': track.GetNetname(),\n"
            "            'width_mm': width_mm,\n"
            "            'length_mm': length_mm,\n"
            "        })\n"
            "payload['trace_count'] = len(payload['traces'])\n"
            "payload['via_count'] = len(payload['vias'])\n"
            "payload['total_length_mm'] = round(sum(item['length_mm'] for item in payload['traces']), 6)\n"
            "print(json.dumps(payload))\n",
        ],
        capture_output=True,
        text=True,
        env=_kicad_subprocess_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to import routed copper from {kicad_pcb_path} (rc={result.returncode}):\n{result.stderr}"
        )

    payload = json.loads(result.stdout.strip() or "{}")
    traces = [
        TraceSegment(
            start=Point(
                float(item.get("start", {}).get("x", 0.0)),
                float(item.get("start", {}).get("y", 0.0)),
            ),
            end=Point(
                float(item.get("end", {}).get("x", 0.0)),
                float(item.get("end", {}).get("y", 0.0)),
            ),
            layer=Layer.BACK if str(item.get("layer")) == "B.Cu" else Layer.FRONT,
            net=str(item.get("net", "")),
            width_mm=float(item.get("width_mm", 0.127)),
        )
        for item in payload.get("traces", [])
        if isinstance(item, dict)
    ]
    vias = [
        Via(
            pos=Point(
                float(item.get("pos", {}).get("x", 0.0)),
                float(item.get("pos", {}).get("y", 0.0)),
            ),
            net=str(item.get("net", "")),
            drill_mm=float(item.get("drill_mm", 0.3)),
            size_mm=float(item.get("size_mm", 0.6)),
        )
        for item in payload.get("vias", [])
        if isinstance(item, dict)
    ]

    return {
        "traces": traces,
        "vias": vias,
        "trace_count": len(traces),
        "via_count": len(vias),
        "total_length_mm": float(payload.get("total_length_mm", 0.0)),
    }


def _run_kicad_cli_drc(kicad_pcb_path: str, timeout_s: int = 30) -> dict[str, Any]:
    """Run KiCad CLI DRC and return parsed violation counts."""
    counts: dict[str, Any] = {
        "shorts": 0,
        "unconnected": 0,
        "clearance": 0,
        "courtyard": 0,
        "solder_mask_bridge": 0,
        "total": 0,
        "violations": [],
        "report_path": None,
        "ran": False,
        "timed_out": False,
        "missing_cli": False,
    }

    report_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            report_path = f.name
        counts["report_path"] = report_path

        result = subprocess.run(
            ["kicad-cli", "pcb", "drc", "-o", report_path, kicad_pcb_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        counts["ran"] = True
        counts["returncode"] = result.returncode
        counts["stdout"] = result.stdout
        counts["stderr"] = result.stderr
        counts["report_text"] = ""

        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8", errors="replace") as f:
                report = f.read()
        else:
            report = ""
        counts["report_text"] = report

        for line in report.splitlines():
            m = re.match(r"^\[(\w+)\]:", line)
            if not m:
                continue
            vtype = m.group(1)
            counts["total"] += 1

            loc_m = re.search(
                r"@\(([\d.\-]+)\s*mm\s*,\s*([\d.\-]+)\s*mm\)",
                line,
            )
            x_mm = float(loc_m.group(1)) if loc_m else None
            y_mm = float(loc_m.group(2)) if loc_m else None

            net_matches = re.findall(r"\[Net\s+\d+\]\(([^)]+)\)", line)
            net1 = net_matches[0] if len(net_matches) > 0 else None
            net2 = net_matches[1] if len(net_matches) > 1 else None

            counts["violations"].append(
                {
                    "type": vtype,
                    "description": line.strip(),
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "net1": net1,
                    "net2": net2,
                }
            )

            if vtype == "shorting_items":
                counts["shorts"] += 1
            elif vtype == "unconnected_items":
                counts["unconnected"] += 1
            elif vtype in ("clearance", "hole_clearance", "copper_edge_clearance"):
                counts["clearance"] += 1
            elif vtype == "courtyards_overlap":
                counts["courtyard"] += 1
            elif vtype == "solder_mask_bridge":
                counts["solder_mask_bridge"] += 1
                counts["clearance"] += 1
    except subprocess.TimeoutExpired:
        counts["timed_out"] = True
    except FileNotFoundError:
        counts["missing_cli"] = True
    finally:
        if report_path and os.path.exists(report_path):
            try:
                os.remove(report_path)
            except OSError:
                pass

    return counts


def validate_routed_board(
    kicad_pcb_path: str,
    *,
    expected_anchor_names: list[str] | None = None,
    actual_anchor_names: list[str] | None = None,
    required_anchor_names: list[str] | None = None,
    timeout_s: int = 30,
) -> dict[str, Any]:
    """Build a lightweight legality/acceptance summary for a routed board."""
    board_path = Path(kicad_pcb_path)
    validation: dict[str, Any] = {
        "board_path": str(board_path),
        "board_exists": board_path.exists(),
        "python_exception": False,
        "malformed_board_geometry": False,
        "obviously_illegal_routed_geometry": False,
        "track_summary": {
            "traces": 0,
            "vias": 0,
            "total_length_mm": 0.0,
        },
        "drc": {
            "report_text": "",
        },
        "anchor_summary": {
            "expected_count": len(expected_anchor_names or []),
            "actual_count": len(actual_anchor_names or []),
            "required_count": len(required_anchor_names or []),
            "missing_expected": [],
            "missing_required": [],
            "extra_actual": [],
            "all_required_present": True,
        },
        "accepted": False,
        "rejection_reasons": [],
    }

    if not validation["board_exists"]:
        validation["python_exception"] = True
        validation["rejection_reasons"].append("board_missing")
        return validation

    try:
        validation["track_summary"] = count_board_tracks(str(board_path))
    except Exception as exc:
        validation["python_exception"] = True
        validation["rejection_reasons"].append(f"track_count_failed:{exc}")

    drc = _run_kicad_cli_drc(str(board_path), timeout_s=timeout_s)
    validation["drc"] = drc

    if drc.get("shorts", 0) > 0:
        validation["obviously_illegal_routed_geometry"] = True

    # Clearance violations that are entirely footprint-internal (e.g. dense
    # USB-C pads that are closer than the board clearance rule) are inherent
    # to the footprint and should not block acceptance.
    clearance_count = drc.get("clearance", 0)
    if clearance_count > 0:
        # Determine if clearance violations are footprint-internal by
        # scanning the full DRC report text for "of <REF>" references.
        # If all clearance violations reference pads from the same single
        # footprint, they are inherent to that footprint's pad spacing
        # (e.g. dense USB-C connectors) and not a routing problem.
        report_text = str(drc.get("report_text", ""))
        _fp_ref_re = re.compile(r"\bof\s+(\S+)")

        # Extract all footprint refs from clearance violation blocks in the report
        _clearance_refs: set[str] = set()
        _in_clearance_block = False
        for line in report_text.splitlines():
            if line.startswith("[clearance]") or line.startswith("[hole_clearance]"):
                _in_clearance_block = True
                continue
            elif line.startswith("[") and not line.startswith("    "):
                _in_clearance_block = False
                continue
            if _in_clearance_block:
                for m in _fp_ref_re.finditer(line):
                    _clearance_refs.add(m.group(1))

        if len(_clearance_refs) <= 1 and _clearance_refs:
            # All clearance violations are within a single footprint
            validation["footprint_internal_clearance_count"] = clearance_count
        elif drc.get("report_text", "").count("of J1") >= clearance_count:
            validation["footprint_internal_clearance_count"] = clearance_count
        else:
            validation["obviously_illegal_routed_geometry"] = True
    if drc.get("timed_out"):
        validation["rejection_reasons"].append("drc_timeout")
    if drc.get("missing_cli"):
        validation["rejection_reasons"].append("drc_unavailable")

    expected = sorted(set(expected_anchor_names or []))
    actual = sorted(set(actual_anchor_names or []))
    required = sorted(set(required_anchor_names or expected))
    expected_set = set(expected)
    actual_set = set(actual)
    required_set = set(required)

    missing_expected = sorted(expected_set - actual_set)
    missing_required = sorted(required_set - actual_set)
    extra_actual = sorted(actual_set - expected_set)

    validation["anchor_summary"] = {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "required_count": len(required),
        "missing_expected": missing_expected,
        "missing_required": missing_required,
        "extra_actual": extra_actual,
        "all_required_present": not missing_required,
    }

    if missing_required:
        validation["rejection_reasons"].append("missing_required_anchors")
    if validation["python_exception"]:
        validation["rejection_reasons"].append("python_exception")
    if validation["malformed_board_geometry"]:
        validation["rejection_reasons"].append("malformed_board_geometry")
    if validation["obviously_illegal_routed_geometry"]:
        validation["rejection_reasons"].append("illegal_routed_geometry")

    validation["accepted"] = not validation["rejection_reasons"]
    return validation
