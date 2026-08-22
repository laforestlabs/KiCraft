"""Router-independent pcbnew, DRC, and routed-copper helpers."""

from __future__ import annotations

import json
import collections
import glob
import math
import os
import re
import shutil
import signal
import site
import subprocess
import sys
import tempfile
import time
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


# Printed by a pcbnew script after its work (incl. board.Save) completes
# successfully. _retry_pcbnew_run uses it to tell a post-work teardown crash
# (pcbnew/wx static-destructor SIGSEGV at interpreter shutdown, AFTER the board
# was saved) apart from a real failure mid-work. See _retry_pcbnew_run.
_PCBNEW_OK_SENTINEL = "__KICRAFT_PCBNEW_OK__"


def run_pcbnew_script(script: str) -> None:
    """Run a pcbnew script string in a fresh subprocess.

    Inline strings are not lintable -- prefer ``run_pcbnew_script_file``
    for any nontrivial workload so import-time errors fire when the
    file is parsed instead of being concealed inside a runtime blob.
    """
    # Emit the success sentinel as the script's last act so a teardown SIGSEGV
    # after a successful Save is not mistaken for a failed operation.
    script = (
        script
        + "\nimport sys as _kicraft_sys\n"
        + f"print({_PCBNEW_OK_SENTINEL!r})\n"
        + "_kicraft_sys.stdout.flush()\n"
    )
    return _retry_pcbnew_run([sys.executable, "-c", script])


def run_pcbnew_script_file(script_path: str, *args: str) -> None:
    """Run a pcbnew script that lives as its own .py file.

    Same SWIG-isolation guarantee as ``run_pcbnew_script`` but the
    script is a real .py file, so type checkers, linters, and IDEs
    can see the pcbnew API calls. The script reads its own argv.
    """
    return _retry_pcbnew_run([sys.executable, str(script_path), *map(str, args)])


def _retry_pcbnew_run(cmd: list[str]) -> None:
    """Shared retry loop for pcbnew subprocess launches.

    The "Failed to load board:" race is real: pcbnew occasionally
    fails to LoadBoard() a file that was just written by another
    pcbnew subprocess if the OS hasn't fully flushed the directory
    entry yet. Up to 6 retries with widening backoff.
    """
    attempts = 6
    delays_s = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
    last_result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(attempts):
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=_kicad_subprocess_env(),
        )
        last_result = result
        if result.returncode == 0:
            return
        # Post-work teardown crash: the script ran to completion and emitted
        # its success sentinel, but the process was then killed by a signal
        # (negative returncode) -- e.g. a pcbnew/wx static-destructor SIGSEGV
        # at interpreter teardown. The board was already saved, so the output
        # is intact; treat it as success rather than failing the whole route.
        # A crash *before* the sentinel (mid-work) has no sentinel in stdout
        # and still falls through to the failure path below.
        stdout = getattr(result, "stdout", "") or ""
        if result.returncode < 0 and _PCBNEW_OK_SENTINEL in stdout:
            print(
                f"warning: pcbnew subprocess completed (success sentinel seen) "
                f"but exited on signal {-result.returncode} during teardown; "
                f"output is intact, treating as success",
                file=sys.stderr,
            )
            return
        stderr = result.stderr or ""
        if "Failed to load board:" not in stderr or attempt == attempts - 1:
            break
        time.sleep(delays_s[min(attempt + 1, len(delays_s) - 1)])

    assert last_result is not None
    stdout = getattr(last_result, "stdout", "") or ""
    raise RuntimeError(
        f"pcbnew subprocess failed (rc={last_result.returncode}):\n"
        f"cmd: {cmd[0]} ... ({len(cmd) - 1} args)\n"
        f"stderr:\n{last_result.stderr}\n"
        + (f"stdout:\n{stdout}" if stdout else "")
    )


def _extract_violation_footprint_refs(
    report_text: str,
    violation_types: set[str],
) -> collections.Counter[str]:
    """Extract footprint refs mentioned inside selected DRC blocks."""
    ref_pattern = re.compile(r"\bof\s+(\S+)")
    ref_counts: collections.Counter[str] = collections.Counter()
    active_block = False
    for line in report_text.splitlines():
        match = re.match(r"\[([^\]]+)\]", line)
        if match:
            active_block = match.group(1) in violation_types
            continue
        if line.startswith("[") and not line.startswith("    "):
            active_block = False
            continue
        if not active_block:
            continue
        for match in ref_pattern.finditer(line):
            ref_counts[match.group(1)] += 1
    return ref_counts


def _extract_clearance_footprint_refs(
    report_text: str,
) -> collections.Counter[str]:
    """Extract footprint refs mentioned inside clearance DRC blocks."""
    return _extract_violation_footprint_refs(
        report_text,
        {"clearance", "hole_clearance"},
    )


def _classify_clearance_violations(
    report_text: str,
    ignorable_refs: set[str] | None = None,
) -> dict[str, int]:
    """Classify each [clearance]/[hole_clearance] block as footprint-internal
    (waivable) or genuine, PER VIOLATION.

    A violation is waivable only when every item line in its block names a
    footprint via "of <REF>" and all named refs are one single footprint
    (pad spacing inherent to a dense footprint, e.g. USB-C), or -- explicit
    per-board escape hatch -- every named ref is in ``ignorable_refs``. A
    violation with any ref-less item (a Track/zone item) involves routed
    copper and is never waivable. The old aggregate-refs approach let
    ref-less track-to-track violations ride along with a single
    footprint-internal one (and double-counted the footprint's mentions).

    Returns ``{"waived": n, "genuine": m}``.
    """
    ignorable = ignorable_refs or set()
    ref_pattern = re.compile(r"\bof\s+(\S+)")
    waived = 0
    genuine = 0
    item_refs: list[set[str]] | None = None  # per-item refs of the open block

    def _close_block() -> None:
        nonlocal waived, genuine, item_refs
        if item_refs is None:
            return
        every_item_named = bool(item_refs) and all(item_refs)
        refs: set[str] = set().union(*item_refs) if item_refs else set()
        if every_item_named and (len(refs) == 1 or refs <= ignorable):
            waived += 1
        else:
            genuine += 1
        item_refs = None

    for line in report_text.splitlines():
        header = re.match(r"\[([^\]]+)\]", line)
        if header:
            _close_block()
            if header.group(1) in ("clearance", "hole_clearance"):
                item_refs = []
            continue
        if (
            item_refs is not None
            and line.startswith("    ")
            and "@(" in line
        ):
            item_refs.append(set(ref_pattern.findall(line)))
    _close_block()
    return {"waived": waived, "genuine": genuine}

def count_board_tracks(kicad_pcb_path: str) -> dict[str, Any]:
    """Count traces, vias, length, and the placed footprints from a board.

    Runs in subprocess to avoid pcbnew SWIG issues. Returns
    ``{traces, vias, total_length_mm, footprints, pads, footprint_refs}``.
    ``footprints`` is the placed-component count and ``footprint_refs`` their
    reference designators -- used to detect an EMPTY board (nothing placed)
    and, at the build gate, a board that silently dropped expected parts. On a
    subprocess failure ``footprints``/``pads`` are ``-1`` (unknown, not empty)
    so callers never misread a count failure as an empty board.
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
            "fps = list(board.GetFootprints())\n"
            "refs = [f.GetReference() for f in fps]\n"
            "pads = sum(len(f.Pads()) for f in fps)\n"
            "print(json.dumps({'traces': traces, 'vias': vias,"
            "  'total_length_mm': round(pcbnew.ToMM(length_nm), 2),"
            "  'footprints': len(fps), 'pads': pads, 'footprint_refs': refs}))\n",
        ],
        capture_output=True,
        text=True,
        env=_kicad_subprocess_env(),
    )
    if result.returncode != 0:
        return {"traces": 0, "vias": 0, "total_length_mm": 0.0,
                "footprints": -1, "pads": -1, "footprint_refs": []}
    return json.loads(result.stdout.strip())


def count_copper_outside_outline(
    kicad_pcb_path: str, tol_mm: float = 0.05
) -> dict[str, Any]:
    """Count track endpoints / via centres lying OUTSIDE the Edge.Cuts outline.

    A router or repair pass can place copper outside the board outline, where even
    copper_edge_clearance never fires. The ``malformed_board_geometry``
    validation flag existed for exactly this class but nothing ever set it
    (2026-07-19 review §2.6). Uses the KiCad-tessellated outline polygon
    (SHAPE_POLY_SET), so circles/arcs/rounded rects are exact; points within
    ``tol_mm`` of the boundary count as inside (float noise guard). Runs in a
    subprocess like every other pcbnew inspection here. ``ok: False`` means
    the outline could not be resolved (malformed Edge.Cuts or tooling
    failure) -- callers must not treat that as "contained".
    """
    script = (
        "import json, pcbnew\n"
        f"board = pcbnew.LoadBoard({kicad_pcb_path!r})\n"
        "poly = pcbnew.SHAPE_POLY_SET()\n"
        "try:\n"
        "    ok = bool(board.GetBoardPolygonOutlines(poly))\n"
        "except Exception:\n"
        "    ok = False\n"
        "if not ok or poly.OutlineCount() == 0:\n"
        "    print(json.dumps({'ok': False, 'outside_tracks': -1,"
        " 'outside_vias': -1, 'examples': []}))\n"
        "    raise SystemExit(0)\n"
        f"tol_nm = int({tol_mm} * 1e6)\n"
        "tol_sq = tol_nm * tol_nm\n"
        "def outside(x, y):\n"
        "    p = pcbnew.VECTOR2I(int(x), int(y))\n"
        "    if poly.Contains(p):\n"
        "        return False\n"
        "    try:\n"
        "        return poly.SquaredDistance(p) > tol_sq\n"
        "    except Exception:\n"
        "        return True\n"
        "ot = ov = 0\n"
        "examples = []\n"
        "for t in board.GetTracks():\n"
        "    if isinstance(t, pcbnew.PCB_VIA):\n"
        "        pos = t.GetPosition()\n"
        "        if outside(pos.x, pos.y):\n"
        "            ov += 1\n"
        "            if len(examples) < 5:\n"
        "                examples.append({'kind': 'via',"
        " 'x_mm': pos.x / 1e6, 'y_mm': pos.y / 1e6,"
        " 'net': t.GetNetname()})\n"
        "    else:\n"
        "        s, e = t.GetStart(), t.GetEnd()\n"
        "        if outside(s.x, s.y) or outside(e.x, e.y):\n"
        "            ot += 1\n"
        "            if len(examples) < 5:\n"
        "                examples.append({'kind': 'track',"
        " 'x_mm': s.x / 1e6, 'y_mm': s.y / 1e6,"
        " 'net': t.GetNetname()})\n"
        "print(json.dumps({'ok': True, 'outside_tracks': ot,"
        " 'outside_vias': ov, 'examples': examples}))\n"
    )
    fallback = {"ok": False, "outside_tracks": -1, "outside_vias": -1, "examples": []}
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=_kicad_subprocess_env(),
    )
    if result.returncode != 0:
        return fallback
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        return fallback


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


def parse_unconnected_nets(report_text: str) -> list[str]:
    """Net names with at least one unconnected (ratsnest) item.

    In a ``kicad-cli pcb drc`` text report each ``[unconnected_items]:`` header
    is followed by indented item lines that name the net in brackets, e.g.::

        [unconnected_items]: Missing connection between items
            @(8.10 mm, 9.78 mm): Pad A4B9 [VBUS] of J1 on F.Cu
            @(8.10 mm, 14.58 mm): Pad B4A9 [VBUS] of J1 on F.Cu

    Returns the de-duplicated nets (order preserved) so callers can tell a
    poured power/ground net (closes on zone fill) from a real signal-net miss.
    """
    nets: list[str] = []
    seen: set[str] = set()
    in_block = False
    for line in report_text.splitlines():
        header = re.match(r"^\[(\w+)\]:", line)
        if header:
            in_block = header.group(1) == "unconnected_items"
            continue
        if not in_block or "@(" not in line:
            continue
        for net in re.findall(r"\[([^\]]+)\]", line):
            # Guard against any "[Net 3]"-style token that is not a net name.
            if re.match(r"(?i)^net\s+\d+$", net):
                continue
            if net not in seen:
                seen.add(net)
                nets.append(net)
    return nets


def run_kicad_cli_drc(kicad_pcb_path: str, timeout_s: int = 30) -> dict[str, Any]:
    """Run KiCad CLI DRC and return parsed violation counts."""
    counts: dict[str, Any] = {
        "shorts": 0,
        "unconnected": 0,
        "unconnected_nets": [],
        "clearance": 0,
        "copper_edge_clearance": 0,
        "courtyard": 0,
        "tracks_crossing": 0,
        "solder_mask_bridge": 0,
        "annular_width": 0,
        "padstack": 0,
        "items_not_allowed": 0,
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
        counts["unconnected_nets"] = parse_unconnected_nets(report)

        current: dict[str, Any] | None = None
        for line in report.splitlines():
            m = re.match(r"^\[(\w+)\]:", line)
            if not m:
                # KiCad reports are block-oriented: the [type] header line
                # carries the rule text, while indented continuation lines carry
                # positions, nets, and the referenced footprint designators.
                if current is not None and line[:1] in (" ", "\t"):
                    if current["x_mm"] is None:
                        loc_m = re.search(
                            r"@\(([\d.\-]+)\s*mm\s*,\s*([\d.\-]+)\s*mm\)",
                            line,
                        )
                        if loc_m:
                            current["x_mm"] = float(loc_m.group(1))
                            current["y_mm"] = float(loc_m.group(2))
                    net_tokens = re.findall(
                        r"\[Net\s+\d+\]\(([^)]+)\)|\[([^\]]+)\]", line
                    )
                    for formatted, bare in net_tokens:
                        net = (formatted or bare).strip()
                        if (not net or re.match(r"(?i)^net\s+\d+$", net)
                                or net.lower() in {"no net", "<no net>"}):
                            continue
                        if current["net1"] is None:
                            current["net1"] = net
                        elif current["net2"] is None and net != current["net1"]:
                            current["net2"] = net
                    refs = re.findall(
                        r"\bFootprint\s+([A-Z][A-Z0-9_-]*)\b|"
                        r"\b(?:of|from)\s+([A-Z][A-Z0-9_-]*)\b",
                        line,
                    )
                    for first, second in refs:
                        ref = first or second
                        if ref and ref not in current["footprint_refs"]:
                            current["footprint_refs"].append(ref)
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
            current = {
                "type": vtype,
                "description": line.strip(),
                "x_mm": x_mm,
                "y_mm": y_mm,
                "net1": net1,
                "net2": net2,
                "footprint_refs": [],
            }
            if len(counts["violations"]) < 120:
                counts["violations"].append(current)

            if vtype == "shorting_items":
                counts["shorts"] += 1
            elif vtype == "unconnected_items":
                counts["unconnected"] += 1
            elif vtype in ("clearance", "hole_clearance"):
                counts["clearance"] += 1
                # KiCad 9 reports copper-to-copper proximity violations as
                # [clearance]. A zero actual clearance is a genuine short.
                actual_m = re.search(r"actual\s+([\d.]+)\s*mm", line)
                if actual_m and float(actual_m.group(1)) <= 0.001:
                    counts["shorts"] += 1
            elif vtype == "copper_edge_clearance":
                counts["copper_edge_clearance"] += 1
            elif vtype == "tracks_crossing":
                counts["tracks_crossing"] += 1
                counts["shorts"] += 1
            elif vtype == "courtyards_overlap":
                counts["courtyard"] += 1
            elif vtype == "solder_mask_bridge":
                counts["solder_mask_bridge"] += 1
            elif vtype == "annular_width":
                counts["annular_width"] += 1
            elif vtype == "padstack":
                counts["padstack"] += 1
            elif vtype == "items_not_allowed":
                counts["items_not_allowed"] += 1

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
    cfg: dict[str, Any] | None = None,
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

    # An accepted board MUST contain placed components. A board with zero
    # footprints is empty (everything dropped) -- it has no shorts and no
    # ratsnest, so it would otherwise sail through the gate looking "clean".
    # ``footprints == -1`` means the count subprocess failed (unknown), which
    # is handled by ``track_count_failed`` above, not treated as empty.
    _fp = validation["track_summary"].get("footprints", -1)
    if _fp is None:
        _fp = -1
    if int(_fp) == 0:
        validation["rejection_reasons"].append("empty_board")

    # Copper containment: routed copper escaping the Edge.Cuts outline
    # regardless of its source. Sets the
    # malformed_board_geometry flag; the reason plumbing below already
    # rejects on it. An unresolved outline is reported but NOT treated as
    # escaped copper -- boards whose Edge.Cuts genuinely fails to close are
    # caught by their own gates.
    if (cfg or {}).get("outline_containment_check", True):
        _containment = count_copper_outside_outline(str(board_path))
        validation["copper_outside_outline"] = _containment
        if _containment.get("ok") and (
            int(_containment.get("outside_tracks", 0) or 0) > 0
            or int(_containment.get("outside_vias", 0) or 0) > 0
        ):
            validation["malformed_board_geometry"] = True

    drc = run_kicad_cli_drc(str(board_path), timeout_s=timeout_s)
    validation["drc"] = drc

    if drc.get("shorts", 0) > 0:
        validation["obviously_illegal_routed_geometry"] = True

    # Clearance violations that are entirely footprint-internal (e.g. dense
    # USB-C pads that are closer than the board clearance rule) are inherent
    # to the footprint and should not block acceptance.
    clearance_count = drc.get("clearance", 0)
    if clearance_count > 0:
        # Classify each clearance violation individually: waive only the
        # blocks whose every item names the same single footprint (pad
        # spacing inherent to a dense footprint, e.g. USB-C). Any block
        # naming routed copper (a ref-less Track/zone item) or spanning
        # two footprints is a genuine routing fault and fails the gate.
        report_text = str(drc.get("report_text", ""))
        drc["clearance_footprint_refs"] = sorted(
            set(_extract_clearance_footprint_refs(report_text))
        )
        ignorable_refs = (
            set(cfg.get("ignorable_footprint_refs", [])) if cfg else set()
        )
        verdict = _classify_clearance_violations(report_text, ignorable_refs)
        if verdict["waived"]:
            validation["footprint_internal_clearance_count"] = verdict["waived"]
        if verdict["genuine"]:
            validation["obviously_illegal_routed_geometry"] = True
    if drc.get("copper_edge_clearance", 0) > 0:
        # Edge-mount connector PADS no longer get a blanket copper_edge waiver.
        # The composer now keeps the board edge a copper-to-edge clearance
        # outboard of an edge-zoned connector's pads (connector_edge_pad_clearance_mm
        # in _repair_parent_outline), so a correctly flush-mounted connector
        # produces no pad-to-edge violation to waive -- the geometry is fixed at
        # the source instead of masked here. Only the explicit per-board
        # ignorable_footprint_refs escape hatch remains; anything else (a stray
        # track near the edge, a genuinely too-close pad) fails loudly.
        report_text = str(drc.get("report_text", ""))
        copper_edge_refs = set(
            _extract_violation_footprint_refs(report_text, {"copper_edge_clearance"})
        )
        drc["copper_edge_footprint_refs"] = sorted(copper_edge_refs)
        ignorable_refs = set(cfg.get("ignorable_footprint_refs", [])) if cfg else set()
        if copper_edge_refs and copper_edge_refs <= ignorable_refs:
            validation["footprint_internal_copper_edge_count"] = int(
                drc.get("copper_edge_clearance", 0)
            )
        else:
            validation["obviously_illegal_routed_geometry"] = True

    # Connector-shield through-hole pads (USB-C shield tabs, etc.) are zero- or
    # low-annular by footprint design, so KiCad reports annular_width / padstack
    # items on them. These are intrinsic to the part, not a routing fault, and
    # are normally waived. Label them as footprint-internal when confined to
    # edge/ignorable connector refs. This does NOT change acceptance -- these
    # types were never blockers -- it just surfaces them as intentionally
    # waived rather than as unexplained DRC noise.
    for _drc_type, _label in (
        ("annular_width", "footprint_internal_annular_count"),
        ("padstack", "footprint_internal_padstack_count"),
    ):
        if drc.get(_drc_type, 0) > 0:
            report_text = str(drc.get("report_text", ""))
            _refs = set(_extract_violation_footprint_refs(report_text, {_drc_type}))
            ignorable_refs = set(cfg.get("ignorable_footprint_refs", [])) if cfg else set()
            edge_component_refs = {
                ref
                for ref, zone in (cfg.get("component_zones", {}) if cfg else {}).items()
                if isinstance(zone, dict) and zone.get("edge")
            }
            if _refs and _refs <= (ignorable_refs | edge_component_refs):
                validation[_label] = int(drc.get(_drc_type, 0))
                validation.setdefault("waived_connector_shield_refs", [])
                validation["waived_connector_shield_refs"] = sorted(
                    set(validation["waived_connector_shield_refs"]) | _refs
                )

    if drc.get("timed_out"):
        validation["rejection_reasons"].append("drc_timeout")
    if drc.get("missing_cli"):
        validation["rejection_reasons"].append("drc_unavailable")
    # kicad-cli is invoked WITHOUT --exit-code-violations, so a nonzero exit
    # means the tool itself failed (crash, bad invocation, unreadable board).
    # When it also reported no violations, every zero count above is vacuous
    # and the board must not read as "clean" (2026-07-19 review §2.3). A
    # nonzero exit WITH parsed violations keeps the parsed verdict -- the
    # per-category gates above already act on it.
    if (
        drc.get("ran")
        and int(drc.get("returncode", 0) or 0) != 0
        and not drc.get("violations")
    ):
        validation["rejection_reasons"].append("drc_failed")

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
