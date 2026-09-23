"""Pinned KiCad Routing Tools adapter with strict copper custody."""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


KICAD_ROUTING_TOOLS_VERSION = "0.20.2"
KICAD_ROUTING_TOOLS_COMMIT = "3ceb773722bea67aa3685e7ee430c0c0d17ef38d"
_KRT_NATIVE_VERSION = "0.20.1"
_KRT_PREFLIGHT_CACHE: dict[tuple[str, str], dict[str, str]] = {}

# Pinned KiCad Routing Tools exposes no NPTH-to-copper floor. Its native
# obstacle map instead hard-codes this value, so a stricter KiCad project rule
# must be expressed as a rule-area before invoking it.
_KRT_NPTH_TRACK_CLEARANCE_MM = 0.20
_KICAD_DEFAULT_MIN_HOLE_CLEARANCE_MM = 0.25
_NPTH_KEEPOUT_ZONE_NAME = "KiCraft NPTH hole clearance"
_NPTH_KEEPOUT_POLYGON_SIDES = 32



class KicadRoutingToolsUnavailableError(RuntimeError):
    """The pinned KiCad Routing Tools runtime is not installed or usable."""


class KicadRoutingToolsTimeoutError(RuntimeError):
    """The router hit this invocation's wall-clock deadline and was killed.

    Deliberately distinct from a router *failure*. A deadline says the search
    did not finish inside the slice it was given; it says nothing about whether
    the board can be routed. Callers must keep it visible (the evidence paths
    ride along in the message) but must NOT read it as geometric infeasibility
    -- that turned "slow" into "structurally unroutable" and aborted whole runs
    (self-eval 2026-09-15, run_10 r1: twelve consecutive 120 s deadlines
    classified as a terminal ``routing_exception``).
    """


class RoutingCopperPreservationError(RuntimeError):
    """KiCadRoutingTools returned a board missing authoritative input copper."""

    def __init__(self, message: str, stats: dict[str, Any]) -> None:
        super().__init__(message)
        self.stats = stats



def _krt_root(config: dict[str, Any]) -> Path:
    raw = config.get("kicad_routing_tools_path", "")
    if not raw and "kicad_routing_tools_path" in config:
        raw = os.environ.get("KICRAFT_KICAD_ROUTING_TOOLS_PATH", "")
    raw = str(raw or "").strip()
    if not raw:
        raise KicadRoutingToolsUnavailableError(
            "KiCadRoutingTools is selected but kicad_routing_tools_path is unset. "
            f"Clone commit {KICAD_ROUTING_TOOLS_COMMIT} and configure its repository path."
        )
    return Path(os.path.expanduser(raw)).resolve()


def preflight_kicad_routing_tools(config: dict[str, Any] | None = None) -> dict[str, str]:
    if config is None:
        from kicraft.autoplacer.config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    root = _krt_root(config)
    raw_python = config.get("kicad_routing_tools_python", sys.executable)
    if (not raw_python and "kicad_routing_tools_python" in config):
        raw_python = os.environ.get(
            "KICRAFT_KICAD_ROUTING_TOOLS_PYTHON", sys.executable
        )
    raw_python = str(raw_python or sys.executable)
    python = shutil.which(os.path.expanduser(raw_python))
    if python is None:
        candidate = Path(os.path.expanduser(raw_python))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            python = str(candidate.absolute())
    if python is None:
        raise KicadRoutingToolsUnavailableError(
            f"KiCadRoutingTools Python interpreter not found: {raw_python}"
        )
    # Keep a virtualenv's executable symlink intact: resolving it to the base
    # interpreter discards the virtualenv dependency context at process launch.
    python = os.path.abspath(python)
    cache_key = (str(root), python)
    cached = _KRT_PREFLIGHT_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    route_script = root / "py_router" / "route.py"
    version_file = root / "VERSION"
    problems: list[str] = []
    if not route_script.is_file():
        problems.append(f"route CLI not found: {route_script}")

    source_version = ""
    if not version_file.is_file():
        problems.append(f"VERSION file not found: {version_file}")
    else:
        try:
            source_version = version_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            problems.append(f"could not read VERSION file {version_file}: {exc}")
        else:
            if source_version != KICAD_ROUTING_TOOLS_VERSION:
                problems.append(
                    f"VERSION is {source_version!r}, "
                    f"expected pinned {KICAD_ROUTING_TOOLS_VERSION!r}"
                )

    revision = ""
    if not (root / ".git").exists():
        problems.append(f"Git checkout metadata not found: {root / '.git'}")
    else:
        try:
            revision_check = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            problems.append(f"could not verify Git checkout revision: {exc}")
        else:
            revision = revision_check.stdout.strip()
            if revision_check.returncode or revision != KICAD_ROUTING_TOOLS_COMMIT:
                detail = revision or revision_check.stderr.strip() or "unknown"
                problems.append(
                    f"checkout is {detail}, "
                    f"expected pinned commit {KICAD_ROUTING_TOOLS_COMMIT}"
                )

    native_version = ""
    try:
        startup_check = subprocess.run(
            [
                python,
                "-c",
                "from py_router.startup_checks import run_all_checks; "
                "print(run_all_checks())",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        problems.append(
            "upstream startup checks failed"
            + (f": {detail}" if detail else f": {exc}")
        )
    else:
        output_lines = [line.strip() for line in startup_check.stdout.splitlines() if line.strip()]
        native_version = output_lines[-1] if output_lines else ""
        if native_version != _KRT_NATIVE_VERSION:
            problems.append(
                f"native router is {native_version or 'unknown'!r}, "
                f"expected pinned {_KRT_NATIVE_VERSION!r}"
            )

    if problems:
        raise KicadRoutingToolsUnavailableError(
            "KiCadRoutingTools backend unavailable:\n  - " + "\n  - ".join(problems)
        )
    result = {
        "backend": "kicad-routing-tools",
        "root": str(root),
        "python": python,
        "version": source_version,
        "commit": revision,
        "native_version": native_version,
    }
    _KRT_PREFLIGHT_CACHE[cache_key] = result
    return dict(result)


def _project_routing_floors(project: Path, config: dict[str, Any]) -> dict[str, float]:
    """Bind every adaptive router escape to the project's actual DRC contract."""
    from kicraft.autoplacer.fab_profile import fab_floors, fanout_via

    body = json.loads(project.read_text(encoding="utf-8"))
    rules = body.get("board", {}).get("design_settings", {}).get("rules", {})
    classes = body.get("net_settings", {}).get("classes", [])
    default = next((item for item in classes if item.get("name") == "Default"), None)
    if default is None:
        raise ValueError(f"Routing project has no Default net class: {project}")

    def number(value: Any) -> float:
        result = float(value)
        if isinstance(value, bool) or not math.isfinite(result) or result < 0:
            raise ValueError(f"Routing project has invalid fabrication constraints: {project}")
        return result

    capability = fab_floors(config)
    via_diameter, via_drill = fanout_via(config)
    via_diameter, via_drill = number(via_diameter), number(via_drill)
    default_clearance = number(default["clearance"])
    class_clearances = [
        default_clearance if item.get("clearance") is None else number(item["clearance"])
        for item in classes
    ]
    requested_ceiling = config.get("kicad_routing_tools_clearance_mm")
    if requested_ceiling is not None and number(requested_ceiling) < max(class_clearances):
        raise ValueError("Router clearance override would weaken a declared net-class clearance")
    clearance = max(
        number(rules.get("min_clearance", 0)),
        *class_clearances,
        number(capability["clearance_mm"]),
    )
    drill = max(number(rules.get("min_through_hole_diameter", 0)), via_drill)
    annular = max(
        number(rules.get("min_via_annular_width", 0)),
        number(rules.get("min_hole_clearance", 0)) - clearance,
        (via_diameter - via_drill) / 2,
    )
    floors = {
        "clearance": clearance,
        "track_width": max(number(rules.get("min_track_width", 0)), number(capability["track_mm"])),
        "via_diameter": max(number(rules.get("min_via_diameter", 0)), via_diameter,
                            drill + 2 * annular),
        "via_drill": drill,
        "annular": annular,
        # These dimensions do not have a reviewed finer escape class: retain
        # the pinned router's 0.2 mm floor even when the project allows less.
        "hole_to_hole": max(number(rules.get("min_hole_to_hole", 0)), 0.2),
        "board_edge": max(number(rules.get("min_copper_edge_clearance", 0)), 0.2),
    }
    return floors


def _krt_command(
    input_path: str, output_path: str, config: dict[str, Any], fab_overrides: Path
) -> list[str]:
    root = _krt_root(config)
    python = (
        config.get("kicad_routing_tools_python")
        or (
            os.environ.get("KICRAFT_KICAD_ROUTING_TOOLS_PYTHON", sys.executable)
            if "kicad_routing_tools_python" in config else sys.executable
        )
    )
    cmd = [
        str(python),
        str(root / "py_router" / "route.py"),
        str(Path(input_path).resolve()),
        str(Path(output_path).resolve()),
        "--nets", "*",
        "--no-fix-drc-settings",
        "--keep-input-copper",
        "--fab-overrides", str(fab_overrides),
        "--max-iterations", str(config.get("kicad_routing_tools_max_iterations", 200000)),
        "--max-ripup", str(config.get("kicad_routing_tools_max_ripup", 3)),
        "--ordering", str(config.get("kicad_routing_tools_ordering", "mps")),
    ]
    clearance = config.get("kicad_routing_tools_clearance_mm")
    if clearance is not None:
        cmd.extend(["--clearance", str(clearance)])
    layers = config.get("kicad_routing_tools_layers")
    if layers:
        cmd.extend(["--layers", *map(str, layers)])
    return cmd


def _fingerprint_multiset(items: list[Any], fingerprint: Any) -> Counter:
    return Counter(fingerprint(item) for item in items)


def _fingerprint_rows(counter: Counter) -> list[dict[str, Any]]:
    return [
        {"fingerprint": list(value), "count": count}
        for value, count in sorted(counter.items(), key=lambda item: repr(item[0]))
    ]


def _preservation_group(
    expected_items: list[Any], actual_items: list[Any], fingerprint: Any
) -> dict[str, Any]:
    expected = _fingerprint_multiset(expected_items, fingerprint)
    actual = _fingerprint_multiset(actual_items, fingerprint)
    matched = expected & actual
    missing = expected - actual
    return {
        "expected_count": sum(expected.values()),
        "matched_count": sum(matched.values()),
        "missing_count": sum(missing.values()),
        "expected": _fingerprint_rows(expected),
        "matched": _fingerprint_rows(matched),
        "missing": _fingerprint_rows(missing),
    }


def _krt_json_summaries(stdout: str, stderr: str) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for stream in (stdout, stderr):
        for line in stream.splitlines():
            if "JSON_SUMMARY:" not in line:
                continue
            try:
                summary = json.loads(line.split("JSON_SUMMARY:", 1)[1].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(summary, dict):
                summaries.append(summary)
    return summaries



def _propagate_sibling_project_rules(src_pcb_path: str, dst_pcb_path: str) -> None:
    """Carry authoritative KiCad project and custom-rule sidecars."""
    src_stem = os.path.splitext(src_pcb_path)[0]
    dst_stem = os.path.splitext(dst_pcb_path)[0]
    for suffix in (".kicad_pro", ".kicad_dru"):
        src_rules = src_stem + suffix
        dst_rules = dst_stem + suffix
        if os.path.isfile(src_rules) and os.path.abspath(src_rules) != os.path.abspath(dst_rules):
            shutil.copy2(src_rules, dst_rules)


def _project_min_hole_clearance(project: Path) -> float:
    """Return KiCad's effective board-level hole-to-copper rule."""
    body = json.loads(project.read_text(encoding="utf-8"))
    value = (
        body.get("board", {})
        .get("design_settings", {})
        .get("rules", {})
        .get("min_hole_clearance", _KICAD_DEFAULT_MIN_HOLE_CLEARANCE_MM)
    )
    result = float(value)
    if isinstance(value, bool) or not math.isfinite(result) or result < 0:
        raise ValueError(f"Routing project has invalid fabrication constraints: {project}")
    return result


def _board_has_npth_pads(board_path: Path) -> bool:
    """Cheaply avoid a pcbnew staging pass on boards that have no NPTH pads."""
    return "np_thru_hole" in board_path.read_text(encoding="utf-8", errors="replace")


def _stamp_npth_keepouts(
    board_path: Path,
    *,
    hole_clearance_mm: float,
    router_clearance_mm: float,
) -> dict[str, int | float]:
    """Stamp temporary, all-copper rule areas that lift KRT's NPTH floor.

    The contour dilates the actual drill capsule by only the clearance KRT
    cannot model. KRT then applies its normal track/via half-width and
    ``router_clearance_mm`` obstacle inflation, yielding the project's required
    copper-edge-to-hole-edge distance for round holes and slots alike.
    """
    from kicraft.autoplacer.routing_board import run_pcbnew_script

    summary_path = board_path.with_suffix(".npth-keepouts.json")
    margin_mm = max(0.0, hole_clearance_mm - router_clearance_mm)
    script = f"""
import json
import math
import pcbnew

_BOARD_PATH = {str(board_path)!r}
_SUMMARY_PATH = {str(summary_path)!r}
_MARGIN_MM = {margin_mm!r}
_ZONE_NAME = {_NPTH_KEEPOUT_ZONE_NAME!r}
_SIDES = {_NPTH_KEEPOUT_POLYGON_SIDES!r}


def _convex_hull(points):
    points = sorted(set(points))
    if len(points) < 3:
        return points

    def cross(origin, a, b):
        return ((a[0] - origin[0]) * (b[1] - origin[1])
                - (a[1] - origin[1]) * (b[0] - origin[0]))

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _drill_capsule_polygon(pad):
    drill = pad.GetDrillSize()
    width = pcbnew.ToMM(drill.x)
    height = pcbnew.ToMM(drill.y)
    if width <= 0 or height <= 0:
        return []
    center = pad.GetPosition()
    cx, cy = pcbnew.ToMM(center.x), pcbnew.ToMM(center.y)
    angle = math.radians(float(pad.GetOrientationDegrees()))
    if width >= height:
        axis = (math.cos(angle), -math.sin(angle))
        length = width - height
        radius = height / 2.0
    else:
        axis = (math.sin(angle), math.cos(angle))
        length = height - width
        radius = width / 2.0
    half_length = length / 2.0
    centers = [
        (cx - axis[0] * half_length, cy - axis[1] * half_length),
        (cx + axis[0] * half_length, cy + axis[1] * half_length),
    ]
    # Circumscribed rather than inscribed: its chord edges cannot cut into the
    # true circular drill wall between vertices.
    outer_radius = (radius + _MARGIN_MM) / math.cos(math.pi / _SIDES)
    points = [
        (
            x + outer_radius * math.cos(2.0 * math.pi * i / _SIDES),
            y + outer_radius * math.sin(2.0 * math.pi * i / _SIDES),
        )
        for x, y in centers
        for i in range(_SIDES)
    ]
    return _convex_hull(points)


board = pcbnew.LoadBoard(_BOARD_PATH)
for zone in list(board.Zones()):
    if zone.GetIsRuleArea() and zone.GetZoneName() == _ZONE_NAME:
        board.Delete(zone)

copper_layers = list(board.GetEnabledLayers().CuStack())

holes = 0
zones = 0
for footprint in board.Footprints():
    for pad in footprint.Pads():
        if pad.GetAttribute() != pcbnew.PAD_ATTRIB_NPTH:
            continue
        polygon = _drill_capsule_polygon(pad)
        if len(polygon) < 3:
            continue
        holes += 1
        for layer in copper_layers:
            zone = pcbnew.ZONE(board)
            zone.SetLayer(layer)
            zone.SetIsRuleArea(True)
            zone.SetDoNotAllowTracks(True)
            zone.SetDoNotAllowVias(True)
            zone.SetDoNotAllowPads(False)
            zone.SetDoNotAllowCopperPour(True)
            zone.SetZoneName(_ZONE_NAME)
            outline = zone.Outline()
            outline.NewOutline()
            for x, y in polygon:
                outline.Append(pcbnew.FromMM(x), pcbnew.FromMM(y))
            board.Add(zone)
            zones += 1

board.BuildConnectivity()
board.Save(_BOARD_PATH)
with open(_SUMMARY_PATH, "w", encoding="utf-8") as out:
    json.dump({{"holes": holes, "zones": zones, "margin_mm": _MARGIN_MM}}, out)
"""
    run_pcbnew_script(script)
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    finally:
        summary_path.unlink(missing_ok=True)


def _strip_npth_keepouts(board_path: Path) -> None:
    """Remove the temporary router-side keepouts from a finished board.

    The rule areas are an input-only workaround: they make KiCadRoutingTools
    respect the project hole rule, and the routed copper they produced already
    satisfies it. A shipped board must carry only authored geometry, while the
    normal DRC gate still measures the real clearance on the routed copper.
    """
    from kicraft.autoplacer.routing_board import run_pcbnew_script

    script = f"""
import pcbnew

_BOARD_PATH = {str(board_path)!r}
_ZONE_NAME = {_NPTH_KEEPOUT_ZONE_NAME!r}

board = pcbnew.LoadBoard(_BOARD_PATH)
removed = 0
for zone in list(board.Zones()):
    if zone.GetIsRuleArea() and zone.GetZoneName() == _ZONE_NAME:
        board.Delete(zone)
        removed += 1
if removed:
    board.BuildConnectivity()
    board.Save(_BOARD_PATH)
print("__KICRAFT_NPTH_STRIPPED__", removed)
"""
    run_pcbnew_script(script)


def _stage_npth_keepouts(
    input_board: Path,
    output_board: Path,
    source_project: Path,
    *,
    hole_clearance_mm: float,
    router_clearance_mm: float,
) -> tuple[Path, dict[str, int | float] | None]:
    """Copy KRT input and stamp it only when the project's rule needs lifting."""
    staged_board = output_board.with_name(
        f"{output_board.stem}.krt-input.kicad_pcb"
    )
    shutil.copy2(input_board, staged_board)
    for suffix in (".kicad_pro", ".kicad_dru"):
        source = source_project.with_suffix(suffix)
        destination = staged_board.with_suffix(suffix)
        if source.is_file():
            shutil.copy2(source, destination)
        else:
            destination.unlink(missing_ok=True)
    if (
        hole_clearance_mm <= max(router_clearance_mm, _KRT_NPTH_TRACK_CLEARANCE_MM)
        or not _board_has_npth_pads(input_board)
    ):
        return staged_board, None
    summary = _stamp_npth_keepouts(
        staged_board,
        hole_clearance_mm=hole_clearance_mm,
        router_clearance_mm=router_clearance_mm,
    )
    # pcbnew.Save rewrites its project sidecar; restore the authoritative rules,
    # just as the normal board-stamping adapter does.
    _propagate_sibling_project_rules(
        str(source_project.with_suffix(".kicad_pcb")), str(staged_board)
    )
    return staged_board, summary

def route_with_kicad_routing_tools(
    kicad_pcb_path: str,
    output_path: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Route directly while preserving input copper and project design rules."""
    from kicraft.autoplacer.brain.copper_accounting import (
        fingerprint_trace,
        fingerprint_via,
    )
    from kicraft.autoplacer.routing_board import import_routed_copper

    input_board = Path(kicad_pcb_path).resolve()
    output_board = Path(output_path).resolve()
    if input_board == output_board:
        raise ValueError(
            "KiCadRoutingTools input and output board paths must be distinct"
        )

    runtime = preflight_kicad_routing_tools(config)
    root = Path(runtime["root"])
    output_board.unlink(missing_ok=True)
    input_copper = import_routed_copper(str(input_board))

    expected_project = input_board.with_suffix(".kicad_pro")
    candidates: list[Path] = []
    configured_board = str(config.get("pcb_path", "") or "").strip()
    if configured_board:
        candidates.append(
            Path(os.path.expanduser(configured_board)).absolute().with_suffix(
                ".kicad_pro"
            )
        )
    candidates.extend(sorted(input_board.parent.glob("*.kicad_pro")))
    source_project = (
        expected_project
        if expected_project.is_file()
        else next((path for path in candidates if path.is_file()), None)
    )
    if source_project is None:
        raise KicadRoutingToolsUnavailableError(
            "KiCadRoutingTools requires a sibling .kicad_pro before routing "
            f"{input_board}"
        )

    floors = _project_routing_floors(source_project, config)
    router_input, npth_keepouts = _stage_npth_keepouts(
        input_board,
        output_board,
        source_project,
        hole_clearance_mm=_project_min_hole_clearance(source_project),
        router_clearance_mm=floors["clearance"],
    )
    fab_overrides = output_board.with_suffix(".fab-overrides.txt")
    fab_overrides.write_text(
        "".join(f"{key} = {value:.9g}\n" for key, value in floors.items()),
        encoding="utf-8",
    )
    command = _krt_command(str(router_input), str(output_board), config, fab_overrides)
    timeout_s = int(config.get("kicad_routing_tools_timeout_s", 120))
    environment = os.environ.copy()
    environment["KICAD_RIP_PREEXISTING"] = "0"
    environment["KICAD_PLANE_FINALIZE"] = "0"
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    timed_out = False
    proc = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
    if output_board.is_file():
        _propagate_sibling_project_rules(str(router_input), str(output_board))
        if npth_keepouts is not None:
            # The router consumed the keepouts; the routed copper already meets
            # the project rule, so the shipped board keeps only authored geometry.
            _strip_npth_keepouts(output_board)
            _propagate_sibling_project_rules(str(router_input), str(output_board))
    elapsed = time.monotonic() - started
    stdout_path = output_board.with_suffix(".router.stdout.log")
    stderr_path = output_board.with_suffix(".router.stderr.log")
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    evidence = f"stdout={stdout_path}; stderr={stderr_path}; rules={fab_overrides}"
    if timed_out:
        raise KicadRoutingToolsTimeoutError(
            f"KiCadRoutingTools timed out after {timeout_s}s; {evidence}"
        )
    if proc.returncode != 0 or not output_board.is_file():
        detail = (stderr or stdout or "no output").strip()[-4000:]
        raise RuntimeError(f"KiCadRoutingTools failed (rc={proc.returncode}): {detail}; {evidence}")
    if not output_board.with_suffix(".kicad_pro").is_file():
        raise KicadRoutingToolsUnavailableError(
            "KiCadRoutingTools requires a sibling .kicad_pro on its routed output; "
            f"could not propagate project rules to {output_board}"
        )

    output_copper = import_routed_copper(str(output_board))
    preservation = {
        "traces": _preservation_group(
            input_copper.get("traces", []),
            output_copper.get("traces", []),
            fingerprint_trace,
        ),
        "vias": _preservation_group(
            input_copper.get("vias", []),
            output_copper.get("vias", []),
            fingerprint_via,
        ),
    }
    preserved = (
        preservation["traces"]["missing_count"] == 0
        and preservation["vias"]["missing_count"] == 0
    )
    json_summaries = _krt_json_summaries(stdout, stderr)
    summary = json_summaries[0] if json_summaries else {}
    stats = {
        "backend": "kicad-routing-tools",
        "version": runtime["version"],
        "commit": runtime["commit"],
        "source_version": runtime["version"],
        "source_commit": runtime["commit"],
        "native_version": runtime["native_version"],
        "source_input_path": str(input_board),
        "router_input_path": str(router_input),
        "npth_hole_clearance_keepouts": npth_keepouts,
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 3),
        "successful_nets": summary.get("successful"),
        "failed_nets": summary.get("failed"),
        "total_vias": summary.get("total_vias"),
        "router_time_s": summary.get("total_time"),
        "json_summaries": json_summaries,
        "input_copper_preservation": preservation,
        "preserved_existing_copper": preserved,
        "command": command,
        "fabrication_floors": floors,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "_raw_stdout": stdout,
        "_raw_stderr": stderr,
    }
    if not preserved:
        raise RoutingCopperPreservationError(
            "KiCadRoutingTools failed to preserve input copper "
            f"(missing traces={preservation['traces']['missing_count']}, "
            f"vias={preservation['vias']['missing_count']}); "
            f"routed output retained at {output_board}",
            stats,
        )
    return stats
