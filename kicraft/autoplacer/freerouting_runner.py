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
import collections
import glob
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


class FreeroutingUnavailableError(RuntimeError):
    """The routing toolchain (Java JRE and/or the FreeRouting jar) is missing.

    Raised up front (and from run_freerouting) so a host without Java or the
    jar fails with one clear, actionable message instead of every leaf quietly
    degrading to a generic ``routing_exception`` and the build ending in the
    misleading "board not routable as placed".
    """


def resolve_java(java_bin: str = "java") -> str | None:
    """Locate a usable ``java`` executable, or return None.

    Resolution order:
      1. ``java_bin`` as an explicit path (absolute or containing a separator)
      2. ``java_bin`` looked up on PATH
      3. common JVM install roots -- ~/.local/lib, /usr/lib/jvm, /opt -- so a
         user-local JRE is found even under the minimal PATH a systemd unit
         runs with.
    """
    jb = os.path.expanduser(java_bin or "java")
    if os.path.sep in jb:
        return jb if os.path.isfile(jb) and os.access(jb, os.X_OK) else None
    found = shutil.which(jb)
    if found:
        return found
    preferred = os.path.expanduser("~/.local/lib/jre/bin/java")
    if os.path.isfile(preferred) and os.access(preferred, os.X_OK):
        return preferred
    candidates: list[str] = []
    for root in (os.path.expanduser("~/.local/lib"), "/usr/lib/jvm", "/opt"):
        candidates.extend(glob.glob(os.path.join(root, "*", "bin", "java")))
        candidates.extend(glob.glob(os.path.join(root, "*", "*", "bin", "java")))
    candidates = [c for c in candidates if os.path.isfile(c) and os.access(c, os.X_OK)]
    return sorted(candidates, reverse=True)[0] if candidates else None


def preflight_routing_toolchain(
    config: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Verify Java + the FreeRouting jar are present before a build routes.

    Returns ``(java_path, jar_path)`` or raises FreeroutingUnavailableError with
    an actionable message naming exactly what to install/set.
    """
    if config is None:
        from kicraft.autoplacer.config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG
    java_bin = config.get("java_bin", "java")
    jar_path = os.path.expanduser(config.get("freerouting_jar", "") or "")
    problems: list[str] = []
    java = resolve_java(java_bin)
    if java is None:
        problems.append(
            f"  - Java runtime not found (java_bin={java_bin!r}; searched PATH, "
            "~/.local/lib, /usr/lib/jvm). Install a JRE "
            "(apt install default-jre-headless) or set 'java_bin' to a JRE path."
        )
    if not jar_path or not os.path.isfile(jar_path):
        problems.append(
            f"  - FreeRouting jar not found at {jar_path or '(unset)'}. Download "
            "freerouting-1.9.0.jar to that path or set 'freerouting_jar'."
        )
    if problems:
        raise FreeroutingUnavailableError(
            "PCB routing toolchain unavailable -- the board cannot be routed:\n"
            + "\n".join(problems)
        )
    return java, jar_path  # type: ignore[return-value]


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


def _run_pcbnew_script(script: str) -> None:
    """Run a pcbnew script string in a fresh subprocess.

    Inline strings are not lintable -- prefer ``_run_pcbnew_script_file``
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


def _run_pcbnew_script_file(script_path: str, *args: str) -> None:
    """Run a pcbnew script that lives as its own .py file.

    Same SWIG-isolation guarantee as ``_run_pcbnew_script`` but the
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


def strip_net_copper(kicad_pcb_path: str, net_name: str) -> None:
    """Remove all tracks/vias and copper zones belonging to a single net.

    Used to clear a net (e.g. GND) so it can be re-handled by one copper plane.
    The leaf-composed GND trace web saturates the signal layer and blocks the
    parent's cross-block signal routing; stripping it lets signals route on a
    clear layer before ground is poured back as a plane.

    Each pass -- track removal, zone removal, connectivity rebuild -- runs in its
    OWN short-lived pcbnew subprocess. Doing all three in a single process
    reliably SIGSEGVs pcbnew on a composed parent board (each pass alone is fine;
    the combination corrupts pcbnew's internal state). One process per pass, each
    reloading from the file the previous one saved, sidesteps the crash.
    """
    load = f"board = pcbnew.LoadBoard({kicad_pcb_path!r})\n"
    save = f"board.Save({kicad_pcb_path!r})\n"
    # 1) tracks/vias on the net
    _run_pcbnew_script(
        "import pcbnew\n" + load + f"net = {net_name!r}\n"
        "for t in list(board.GetTracks()):\n"
        "    if t.GetNetname() == net: board.Remove(t)\n" + save
    )
    # 2) copper zones on the net
    _run_pcbnew_script(
        "import pcbnew\n" + load + f"net = {net_name!r}\n"
        "for z in list(board.Zones()):\n"
        "    if z.GetNetname() == net: board.Remove(z)\n" + save
    )
    # 3) rebuild connectivity from the trimmed board
    _run_pcbnew_script("import pcbnew\n" + load + "board.BuildConnectivity()\n" + save)


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


def min_intra_footprint_pad_gap_mm(kicad_pcb_path: str) -> float | None:
    """Smallest edge-to-edge gap between two *different-net* pads of the same
    footprint, in mm.

    Fine-pitch connectors (USB-C, board-to-board) carry pad gaps below the
    default routing clearance; the autorouter then cannot escape a trace from
    the pad field and the gaps show up as clearance DRC violations. This is
    the signal used to lower the routing clearance for such boards.

    Same-net pad pairs are skipped (they connect anyway, so their proximity is
    not a routing constraint). Returns None when no footprint has two
    different-net pads, or when pcbnew is unavailable.

    Runs in subprocess to avoid pcbnew SWIG issues.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, pcbnew\n"
            f"board = pcbnew.LoadBoard({kicad_pcb_path!r})\n"
            "best = None\n"
            "for fp in board.GetFootprints():\n"
            "    pads = list(fp.Pads())\n"
            "    for i in range(len(pads)):\n"
            "        ba = pads[i].GetBoundingBox()\n"
            "        na = pads[i].GetNetname()\n"
            "        for j in range(i + 1, len(pads)):\n"
            "            nb = pads[j].GetNetname()\n"
            "            if na and na == nb:\n"
            "                continue\n"
            "            bb = pads[j].GetBoundingBox()\n"
            "            dx = max(0, ba.GetLeft() - bb.GetRight(), bb.GetLeft() - ba.GetRight())\n"
            "            dy = max(0, ba.GetTop() - bb.GetBottom(), bb.GetTop() - ba.GetBottom())\n"
            "            g = (dx * dx + dy * dy) ** 0.5\n"
            "            if best is None or g < best:\n"
            "                best = g\n"
            "print(json.dumps({'gap_mm': None if best is None else round(pcbnew.ToMM(int(best)), 4)}))\n",
        ],
        capture_output=True,
        text=True,
        env=_kicad_subprocess_env(),
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip()).get("gap_mm")
    except (json.JSONDecodeError, AttributeError):
        return None


def export_dsn(
    kicad_pcb_path: str,
    dsn_path: str,
    lock_existing_traces: bool = False,
    *,
    target_clearance_um: int | None = None,
    target_width_um: int | None = None,
) -> None:
    """Export Specctra DSN from a KiCad PCB file using pcbnew API.

    Assumes copper zones have already been stripped so FreeRouting starts
    from a clean board containing only footprints, nets, and board geometry.

    If lock_existing_traces is True, all existing tracks and vias are marked
    as locked before export so FreeRouting treats them as fixed pre-routes.

    ``target_clearance_um`` / ``target_width_um`` (micrometres) override the
    routing rule for fine-pitch boards -- see :func:`_patch_dsn_clearance`.
    """
    lock_script = ""
    if lock_existing_traces:
        lock_script = (
            "# Lock all existing traces so FreeRouting treats them as fixed\n"
            "for track in board.GetTracks():\n"
            "    track.SetLocked(True)\n"
        )
    netclass_json = dsn_path + ".netclasses.json"
    _run_pcbnew_script(
        "import pcbnew, json\n"
        f"board = pcbnew.LoadBoard({kicad_pcb_path!r})\n"
        + lock_script
        + "board.BuildConnectivity()\n"
        f"board.Save({kicad_pcb_path!r})\n"
        f"pcbnew.ExportSpecctraDSN(board, {dsn_path!r})\n"
        # Capture netclass clearances + net->class so _inject_netclass_clearances
        # can restore the per-class rules KiCad's DSN export drops (it lumps every
        # net into one kicad_default class at the default clearance).
        "_info = {'classes': {}, 'net_class': {}}\n"
        "for _n, _nc in board.GetAllNetClasses().items():\n"
        "    try:\n"
        "        _info['classes'][str(_n)] = int(round(_nc.GetClearance() / 1000))\n"
        "    except Exception:\n"
        "        pass\n"
        "for _net in board.GetNetInfo().NetsByName().values():\n"
        "    _nm = _net.GetNetname()\n"
        "    if _nm:\n"
        "        _info['net_class'][_nm] = str(_net.GetNetClassName())\n"
        f"open({netclass_json!r}, 'w').write(json.dumps(_info))\n"
    )
    _patch_dsn_clearance(
        dsn_path,
        target_clearance_um=target_clearance_um,
        target_width_um=target_width_um,
    )
    _inject_netclass_clearances(dsn_path)


def _patch_dsn_clearance(
    dsn_path: str,
    *,
    target_clearance_um: int | None = None,
    target_width_um: int | None = None,
) -> None:
    """Normalize the routing rule in a Specctra DSN before FreeRouting.

    Two modes:

    * **Fine-pitch (lower)** -- when ``target_clearance_um`` is set and is
      below the DSN's global clearance, LOWER the global + class clearance to
      it (and, if ``target_width_um`` is given, lower the rule track width) so
      the autorouter can escape dense pad fields (USB-C etc.). Already-tighter
      typed clearances (e.g. KiCad's ``smd_smd`` 0.05 mm export) are left
      untouched -- raising them is exactly what blocks fine-pitch escape.

    * **Legacy (raise)** -- otherwise, raise every type-specific clearance up
      to the global value. KiCad exports reduced clearances for certain types
      (``smd_smd`` at 0.05 mm, etc.); on a normal board those under-cut the
      design rule, so we bring them up to the global clearance.
    """
    with open(dsn_path) as f:
        content = f.read()
    # Global clearance = first bare clearance token (no type qualifier).
    m = re.search(r"\(clearance\s+(\d+)\)", content)
    if not m:
        return
    global_clearance = int(m.group(1))

    if target_clearance_um is not None and target_clearance_um < global_clearance:
        tc = int(target_clearance_um)
        # Lower every bare global clearance (structure rule + class rule).
        content = re.sub(r"\(clearance\s+\d+\)", f"(clearance {tc})", content)
        # Typed clearances: only ever lower, never raise above the target.
        content = re.sub(
            r"\(clearance\s+(\d+)\s+\(type\s+(\w+)\)\)",
            lambda mm: f"(clearance {min(int(mm.group(1)), tc)} (type {mm.group(2)}))",
            content,
        )
        if target_width_um is not None:
            tw = int(target_width_um)
            content = re.sub(
                r"\(width\s+(\d+)\)",
                lambda mm: f"(width {min(int(mm.group(1)), tw)})",
                content,
            )
    else:
        content = re.sub(
            r"\(clearance\s+\d+\s+\(type\s+(\w+)\)\)",
            lambda mm: f"(clearance {global_clearance} (type {mm.group(1)}))",
            content,
        )

    with open(dsn_path, "w") as f:
        f.write(content)


def _split_dsn_tokens(s: str) -> list[str]:
    """Split a DSN token run, keeping ``"quoted net names"`` intact."""
    return re.findall(r'"[^"]*"|\S+', s)


def _inject_netclass_clearances(dsn_path: str) -> None:
    """Restore per-netclass clearances that KiCad's Specctra DSN export drops.

    ``pcbnew.ExportSpecctraDSN`` lumps every net into a single ``kicad_default``
    class at the board default clearance, discarding wider netclass rules (e.g. a
    0.3 mm Power class). FreeRouting then routes those nets too tight and the
    post-route DRC -- which validates against the real netclass rule -- rejects
    the board (``illegal_routed_geometry``). This re-splits the single DSN class
    into one class per board netclass, raising each to
    ``max(dsn_default, netclass_clearance)`` so wider classes (power) are honored
    while nothing routes tighter than before. It runs after
    :func:`_patch_dsn_clearance`, so the default it reads already reflects any
    fine-pitch lowering.

    Best-effort: reads the ``<dsn>.netclasses.json`` sidecar written by
    :func:`export_dsn`; any missing data or parse failure leaves the DSN as-is.
    """
    sidecar = dsn_path + ".netclasses.json"
    try:
        with open(sidecar) as f:
            info = json.load(f)
    except Exception:
        return
    finally:
        try:
            os.remove(sidecar)
        except OSError:
            pass
    classes_um = {str(k): int(v) for k, v in (info.get("classes") or {}).items()}
    net_class = {str(k): str(v) for k, v in (info.get("net_class") or {}).items()}
    if not classes_um or not net_class:
        return

    try:
        with open(dsn_path) as f:
            content = f.read()
        start = content.find("(class kicad_default")
        if start < 0:
            return
        # Balanced-paren scan for the end of this (class ...) block.
        depth, i = 0, start
        while i < len(content):
            if content[i] == "(":
                depth += 1
            elif content[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            return
        block = content[start : i + 1]
        hdr_end = block.find("(", len("(class"))
        if hdr_end < 0:
            return
        # tokens after the class name are the member nets
        nets = _split_dsn_tokens(block[len("(class") : hdr_end])[1:]
        rest = block[hdr_end:-1]
        rule_start = rest.rfind("(rule")
        if rule_start < 0:
            return
        circuit = rest[:rule_start].strip()  # (circuit (use_via ...)) -- preserved
        rule_text = rest[rule_start:]
        mclr = re.search(r"\(clearance\s+(\d+)\)", rule_text)
        if not mclr:
            return
        default_um = int(mclr.group(1))
        mwid = re.search(r"\(width\s+(\d+)\)", rule_text)
        width = mwid.group(1) if mwid else "200"

        groups: dict[str, dict[str, Any]] = {}
        order = 0
        for tok in nets:
            name = tok[1:-1] if len(tok) >= 2 and tok[0] == '"' == tok[-1] else tok
            cls = net_class.get(name)
            if cls is None:
                cls_token, um = "kicad_default", default_um
            else:
                cls_token = re.sub(r"[^A-Za-z0-9_]+", "_", cls) or "class"
                um = max(default_um, classes_um.get(cls, default_um))
            g = groups.get(cls_token)
            if g is None:
                groups[cls_token] = {"nets": [tok], "um": um, "order": order}
                order += 1
            else:
                g["nets"].append(tok)
        # Nothing wider than the default -> leave the DSN byte-for-byte unchanged.
        if all(g["um"] == default_um for g in groups.values()):
            return

        blocks = []
        for cls_token, g in sorted(groups.items(), key=lambda kv: kv[1]["order"]):
            nets_str = " ".join(g["nets"])
            circ = f"\n      {circuit}" if circuit else ""
            blocks.append(
                f"(class {cls_token} {nets_str}{circ}\n"
                f"      (rule\n        (width {width})\n"
                f"        (clearance {g['um']})\n      )\n    )"
            )
        content = content[:start] + "\n  ".join(blocks) + content[i + 1 :]
        with open(dsn_path, "w") as f:
            f.write(content)
    except Exception as exc:  # noqa: BLE001 -- never break routing over this
        print(f"  warning: netclass clearance injection skipped: {exc}")


def _propagate_sibling_pro(src_pcb_path: str, dst_pcb_path: str) -> None:
    """Copy ``src``'s sibling ``.kicad_pro`` onto ``dst``'s, when present.

    pcbnew's ``board.Save()`` emits a *default* sidecar ``.kicad_pro`` (Default
    netclass 0.20 mm), dropping the project's real netclasses. Carrying the
    source project forward keeps the real netclass clearances/patterns on a
    freshly-written board so post-route DRC validates against the same rules
    FreeRouting was given. Best-effort.
    """
    src_pro = os.path.splitext(src_pcb_path)[0] + ".kicad_pro"
    dst_pro = os.path.splitext(dst_pcb_path)[0] + ".kicad_pro"
    try:
        if os.path.isfile(src_pro) and os.path.abspath(src_pro) != os.path.abspath(
            dst_pro
        ):
            shutil.copy2(src_pro, dst_pro)
    except OSError:
        pass


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


_XVFB_WARNED = False


def run_freerouting(
    dsn_path: str,
    ses_path: str,
    jar_path: str,
    timeout_s: int = 120,
    max_passes: int = 40,
    work_dir: str | None = None,
    hide_window: bool = True,
    java_bin: str = "java",
) -> dict[str, Any]:
    """Run FreeRouting CLI and return result metadata.

    Uses start_new_session so the Java process gets its own process group,
    allowing clean kill via os.killpg() on timeout or stop request.

    When `hide_window` is True:
      - FR 2.x: appends --gui.enabled=false + --router.max_passes=<N>.
        The legacy -mp is ignored in 2.x CLI mode.
      - FR 1.x: wraps the java invocation in `xvfb-run -a` so the Swing
        window draws into a virtual framebuffer instead of the real
        display. Requires xorg-x11-server-Xvfb; if xvfb-run isn't on
        PATH, emits a one-time warning and falls back to showing the
        window.
    """
    jar_path = os.path.expanduser(jar_path)
    java = resolve_java(java_bin)
    if java is None:
        raise FreeroutingUnavailableError(
            f"Java runtime not found (java_bin={java_bin!r}); cannot launch "
            "FreeRouting. Install a JRE or set 'java_bin' to a JRE path."
        )
    if not os.path.isfile(jar_path):
        raise FreeroutingUnavailableError(
            f"FreeRouting jar not found at {jar_path!r}; download "
            "freerouting-1.9.0.jar there or set 'freerouting_jar'."
        )
    java_cmd = [
        java,
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

    jar_basename = os.path.basename(jar_path).lower()
    is_v2_plus = "freerouting-2" in jar_basename or "freerouting-3" in jar_basename
    cmd = java_cmd
    if hide_window and is_v2_plus:
        cmd.append("--gui.enabled=false")
        cmd.append(f"--router.max_passes={int(max_passes)}")
    elif hide_window and not is_v2_plus:
        xvfb = shutil.which("xvfb-run")
        if xvfb:
            cmd = [
                xvfb,
                "-a",
                "--server-args=-screen 0 1024x768x16",
            ] + java_cmd
        else:
            global _XVFB_WARNED
            if not _XVFB_WARNED:
                print(
                    "[freerouting] hide_window requested but xvfb-run not on "
                    "PATH. Install xorg-x11-server-Xvfb to suppress the FR "
                    "window on FR 1.x; falling back to windowed mode.",
                    flush=True,
                )
                _XVFB_WARNED = True

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


def _resolve_fine_pitch_rule(
    kicad_pcb_path: str, config: dict[str, Any]
) -> tuple[int | None, int | None]:
    """Decide the FreeRouting clearance / track-width override for a board.

    Returns ``(clearance_um, width_um)`` where each is ``None`` to leave the
    DSN's default rule alone. An explicit ``freerouting_clearance_mm`` forces
    the clearance; otherwise the densest different-net pad gap on the board
    drives auto-detection: if it is tighter than a normal clearance, the
    routing clearance is lowered to clear it -- floored at
    ``freerouting_min_clearance_mm`` for fab safety -- and the track width is
    reduced to ``freerouting_fine_pitch_track_mm`` so a trace can escape.
    """
    normal_clearance_mm = 0.2  # board/DSN default the override compares against
    floor_mm = float(config.get("freerouting_min_clearance_mm", 0.1))
    fine_track_mm = float(config.get("freerouting_fine_pitch_track_mm", 0.15))

    override = config.get("freerouting_clearance_mm")
    target_clearance_mm: float | None = None
    gap_mm: float | None = None
    if override is not None:
        target_clearance_mm = float(override)
    else:
        try:
            gap_mm = min_intra_footprint_pad_gap_mm(kicad_pcb_path)
        except Exception:  # noqa: BLE001 -- detection is best-effort
            gap_mm = None
        if gap_mm is not None and gap_mm < normal_clearance_mm:
            target_clearance_mm = max(floor_mm, gap_mm)

    if target_clearance_mm is None:
        return (None, None)

    target_track_mm = min(normal_clearance_mm, fine_track_mm)
    why = (
        f"override {target_clearance_mm:.3f} mm"
        if override is not None
        else f"min pad gap {gap_mm} mm"
    )
    print(
        f"  fine-pitch routing rule ({why}): clearance "
        f"{target_clearance_mm:.3f} mm, track {target_track_mm:.3f} mm"
    )
    return (int(round(target_clearance_mm * 1000)), int(round(target_track_mm * 1000)))


def _set_board_clearance_um(kicad_pcb_path: str, clearance_um: int) -> None:
    """Cap every netclass clearance on a board at ``clearance_um`` (micrometres).

    The fine-pitch rule lowers FreeRouting's clearance GLOBALLY (so traces can
    escape dense pad fields), but the routed board still declares its original,
    wider default clearance. KiCad DRC then flags every trace routed tighter than
    that wider rule -- failing the geometry acceptance gate on a board that is
    fab-clean at the clearance it was actually built to. Bringing the board's
    netclass clearance down to the routed value makes DRC validate against the
    same rule FreeRouting used. Only lowers (``min``), never widens, so an
    intentionally tighter class is left alone. Best-effort: on failure the board
    keeps its old rule (the prior behavior), so this never breaks routing."""
    cl_nm = int(round(clearance_um)) * 1000  # micrometres -> nanometres
    script = (
        "import pcbnew\n"
        f"b = pcbnew.LoadBoard({kicad_pcb_path!r})\n"
        f"cl = {cl_nm}\n"
        "ns = b.GetDesignSettings().m_NetSettings\n"
        "nc = ns.GetDefaultNetclass()\n"
        "nc.SetClearance(min(nc.GetClearance(), cl))\n"
        "for _name, _nc in b.GetAllNetClasses().items():\n"
        "    _nc.SetClearance(min(_nc.GetClearance(), cl))\n"
        f"b.Save({kicad_pcb_path!r})\n"
    )
    try:
        subprocess.run(
            [sys.executable, "-c", script],
            check=True, capture_output=True, text=True,
        )
    except Exception as exc:  # noqa: BLE001 -- consistency is best-effort
        print(f"  warning: could not set board clearance to {clearance_um} um: {exc}")


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
    hide_window = bool(config.get("freerouting_hide_window", True))

    # Fine-pitch clearance handling: the DSN inherits the board's default
    # clearance (0.2 mm), which is wider than a dense connector's pad gaps, so
    # the autorouter cannot escape its pad field. Detect that case and lower
    # the routing clearance (and track width) to a fab-safe floor that clears
    # the densest part. An explicit `freerouting_clearance_mm` overrides the
    # auto-detection. See _patch_dsn_clearance.
    target_clearance_um, target_width_um = _resolve_fine_pitch_rule(
        kicad_pcb_path, config
    )

    for attempt in range(2):
        with tempfile.TemporaryDirectory() as tmpdir:
            dsn_path = os.path.join(tmpdir, "board.dsn")
            ses_path = os.path.join(tmpdir, "board.ses")

            export_dsn(
                kicad_pcb_path,
                dsn_path,
                lock_existing_traces=preserve_existing_copper,
                target_clearance_um=target_clearance_um,
                target_width_um=target_width_um,
            )

            passes = max_passes if attempt == 0 else max(10, max_passes // 2)
            stats = run_freerouting(
                dsn_path,
                ses_path,
                jar_path,
                timeout_s=timeout_s,
                max_passes=passes,
                hide_window=hide_window,
                java_bin=config.get("java_bin", "java"),
            )
            stats["preserved_existing_copper"] = preserve_existing_copper
            stats["cleared_zones_before_export"] = clear_existing_zones

            if os.path.exists(ses_path):
                import_ses(kicad_pcb_path, ses_path, output_path)
                # import_ses saved a fresh board -> default sidecar .kicad_pro;
                # carry the input board's project (real netclasses) forward so
                # post-route DRC validates against the rules FR actually used.
                _propagate_sibling_pro(kicad_pcb_path, output_path)
                if preserve_existing_copper:
                    _unlock_traces(output_path)
                if target_clearance_um is not None:
                    # The fine-pitch lower routed the board at a tighter, fab-safe
                    # clearance; make the board declare it so KiCad DRC validates
                    # against the same rule instead of the original wider default.
                    _set_board_clearance_um(output_path, target_clearance_um)
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


def _parse_unconnected_nets(report_text: str) -> list[str]:
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


def _run_kicad_cli_drc(kicad_pcb_path: str, timeout_s: int = 30) -> dict[str, Any]:
    """Run KiCad CLI DRC and return parsed violation counts."""
    counts: dict[str, Any] = {
        "shorts": 0,
        "unconnected": 0,
        "unconnected_nets": [],
        "clearance": 0,
        "copper_edge_clearance": 0,
        "courtyard": 0,
        "solder_mask_bridge": 0,
        "annular_width": 0,
        "padstack": 0,
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
        counts["unconnected_nets"] = _parse_unconnected_nets(report)

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
            elif vtype in ("clearance", "hole_clearance"):
                counts["clearance"] += 1
            elif vtype == "copper_edge_clearance":
                counts["copper_edge_clearance"] += 1
            elif vtype == "courtyards_overlap":
                counts["courtyard"] += 1
            elif vtype == "solder_mask_bridge":
                counts["solder_mask_bridge"] += 1
            elif vtype == "annular_width":
                counts["annular_width"] += 1
            elif vtype == "padstack":
                counts["padstack"] += 1
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
        _clearance_ref_counts = _extract_clearance_footprint_refs(report_text)
        _clearance_refs = set(_clearance_ref_counts)
        drc["clearance_footprint_refs"] = sorted(_clearance_refs)

        if len(_clearance_refs) <= 1 and _clearance_refs:
            # All clearance violations are within a single footprint
            validation["footprint_internal_clearance_count"] = clearance_count
        elif _clearance_ref_counts:
            dominant_ref, dominant_count = _clearance_ref_counts.most_common(1)[0]
            ignorable_refs = set(cfg.get("ignorable_footprint_refs", [])) if cfg else set()
            if dominant_count >= clearance_count or dominant_ref in ignorable_refs:
                validation["footprint_internal_clearance_count"] = clearance_count
            else:
                validation["obviously_illegal_routed_geometry"] = True
        elif cfg and clearance_count <= 10 and cfg.get("ignorable_footprint_refs"):
            validation["footprint_internal_clearance_count"] = clearance_count
        else:
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
