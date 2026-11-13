"""Pinned KiCad Routing Tools adapter with strict copper custody."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from kicraft.pcb_layers import OUTER_COPPER_LAYERS, declared_copper_layers


KICAD_ROUTING_TOOLS_VERSION = "0.20.2"
KICAD_ROUTING_TOOLS_COMMIT = "8eeee55fd4ba4ed600b11bd8214de9994e8f5396"
_KRT_NATIVE_VERSION = "0.20.1"
_KRT_PREFLIGHT_CACHE: dict[tuple[str, str], dict[str, str]] = {}

# Pinned KiCad Routing Tools exposes no NPTH-to-copper floor. Its native
# obstacle map instead hard-codes this value, so a stricter KiCad project rule
# must be expressed as a rule-area before invoking it.
_KRT_NPTH_TRACK_CLEARANCE_MM = 0.20
_KICAD_DEFAULT_MIN_HOLE_CLEARANCE_MM = 0.25
_NPTH_KEEPOUT_ZONE_NAME = "KiCraft NPTH hole clearance"
_NPTH_KEEPOUT_POLYGON_SIDES = 32
_PAD_VIA_KEEPOUT_ZONE_NAME = "KiCraft assembly pad via exclusion"




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
    input_path: str, output_path: str, config: dict[str, Any], fab_overrides: Path,
    *, nets: list[str] | None = None, track_width: float | None = None,
    same_net_pad_clearance: float | None = None,
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
        "--nets", *(nets if nets is not None else ["*"]),
        "--no-fix-drc-settings",
        "--keep-input-copper",
        "--fab-overrides", str(fab_overrides),
        "--max-iterations", str(config.get("kicad_routing_tools_max_iterations", 200000)),
        "--max-ripup", str(config.get("kicad_routing_tools_max_ripup", 3)),
        "--ordering", str(config.get("kicad_routing_tools_ordering", "mps")),
    ]
    if track_width is not None:
        cmd.extend(["--track-width", str(track_width)])
    if same_net_pad_clearance is not None:
        cmd.extend(["--same-net-pad-clearance", str(same_net_pad_clearance)])
    clearance = config.get("kicad_routing_tools_clearance_mm")
    if clearance is not None:
        cmd.extend(["--clearance", str(clearance)])
    grid_step = config.get("kicad_routing_tools_grid_step_mm")
    if grid_step is not None:
        # Escape fix for pads boxed in at the 0.1 mm default -- see the key's
        # comment in config.py for the measured before/after.
        cmd.extend(["--grid-step", str(grid_step)])
    layers = config.get("kicad_routing_tools_layers")
    if layers:
        cmd.extend(["--layers", *map(str, layers)])
    elif len(declared_copper_layers(input_path)) > len(OUTER_COPPER_LAYERS):
        # A board with inner copper layers keeps its planes inside and its signals on the
        # outer pair: KiCraft's copper bookkeeping carries front/back only (routing_board
        # import + the leaf/parent stampers rebuild tracks from a two-valued layer), so a
        # signal the router placed on an inner layer would come back as front copper and be
        # rebuilt there. An explicit kicad_routing_tools_layers still wins.
        cmd.extend(["--layers", *OUTER_COPPER_LAYERS])
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


# ``output_writer`` prints this when it drops original-file segments that the
# cleanup passes (dead-end sweep #84, orphan islands #217, dangle trim #347)
# proved are not load-bearing: the segment reaches no pad of its net and no
# continuation, so removing it cannot disconnect anything. It is emitted even
# under ``--keep-input-copper`` (that flag protects input copper from *rip-up*,
# not from this cleanup). Parsing the router's own accounting -- rather than
# re-deriving dead ends from geometry, which the net-less trace fingerprint
# cannot support -- lets the custody check distinguish "the router destroyed
# functional input copper" from "the router swept declared unloaded stubs".
_DECLARED_INPUT_SEGMENT_STRIP_RE = re.compile(
    r"Removed (\d+) dead-end input segment\(s\) from the output"
)


def _declared_input_segment_removals(stdout: str) -> int:
    """Original-file segments KRT's cleanup declared it stripped this run."""
    return sum(int(match) for match in _DECLARED_INPUT_SEGMENT_STRIP_RE.findall(stdout))


def _propagate_sibling_project_rules(src_pcb_path: str, dst_pcb_path: str) -> None:
    """Carry authoritative KiCad project and custom-rule sidecars."""
    src_stem = os.path.splitext(src_pcb_path)[0]
    dst_stem = os.path.splitext(dst_pcb_path)[0]
    for suffix in (".kicad_pro", ".kicad_dru"):
        src_rules = src_stem + suffix
        dst_rules = dst_stem + suffix
        if os.path.abspath(src_rules) == os.path.abspath(dst_rules):
            continue
        if os.path.isfile(src_rules):
            shutil.copy2(src_rules, dst_rules)
        else:
            Path(dst_rules).unlink(missing_ok=True)


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

def _stamp_pad_via_keepouts(board_path: Path, zone_name: str) -> dict[str, int]:
    """Expose assembly mask openings to KRT as temporary via-only rule areas.

    KRT inflates rule areas by its via copper radius plus routing clearance,
    which is at least the drill radius. Thus a drill cannot intersect an
    opening even when its centre lies outside the pad. Tracks remain allowed.
    The reviewed manufacturer thermal lands are already handled by ground
    finishing and are the only pads excluded from this ordinary-pad policy.
    """
    from kicraft.autoplacer.routing_board import run_pcbnew_script

    summary_path = board_path.with_suffix(".pad-via-keepouts.json")
    script = f"""
import json
import pcbnew
import sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})
from kicraft.autoplacer.brain.gnd_pour import _authorized_thermal_pad

board = pcbnew.LoadBoard({str(board_path)!r})
name = {zone_name!r}
for zone in list(board.Zones()):
    if zone.GetIsRuleArea() and zone.GetZoneName() == name:
        board.Delete(zone)

pads = zones = authorized = 0
for fp in board.Footprints():
    for pad in fp.Pads():
        if pad.GetAttribute() not in (
            pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN, pcbnew.PAD_ATTRIB_PTH
        ):
            continue
        if _authorized_thermal_pad(fp, pad):
            authorized += 1
            continue
        openings = []
        for copper, mask in ((pcbnew.F_Cu, pcbnew.F_Mask), (pcbnew.B_Cu, pcbnew.B_Mask)):
            if not pad.IsOnLayer(mask):
                continue
            shape = pcbnew.SHAPE_POLY_SET()
            # Transform the actual land shape, orientation, and offset. A
            # negative mask expansion may shrink the opening; retaining the
            # copper outline is conservative in that case.
            expansion = max(0, pad.GetSolderMaskExpansion(mask))
            pad.TransformShapeToPolygon(
                shape, copper, expansion, pcbnew.FromMM(0.001), pcbnew.ERROR_OUTSIDE
            )
            if shape.OutlineCount() == 0:
                raise RuntimeError(
                    "Cannot construct mask opening for " + fp.GetReference() + "." + pad.GetNumber()
                )
            for index in range(shape.OutlineCount()):
                contour = shape.Outline(index)
                # CPoint borrows storage from shape, which is replaced for
                # the next face. Retain coordinates, never borrowed vectors.
                points = []
                for point_index in range(contour.PointCount()):
                    point = contour.CPoint(point_index)
                    points.append((point.x, point.y))
                if len(points) >= 3:
                    openings.append(points)
        if openings:
            pads += 1
        for points in openings:
            zone = pcbnew.ZONE(board)
            # KRT's via obstacle map is global even for a single copper layer.
            zone.SetLayer(pcbnew.F_Cu)
            zone.SetIsRuleArea(True)
            zone.SetDoNotAllowTracks(False)
            zone.SetDoNotAllowVias(True)
            zone.SetDoNotAllowPads(False)
            zone.SetDoNotAllowCopperPour(False)
            zone.SetZoneName(name)
            outline = zone.Outline()
            outline.NewOutline()
            for x, y in points:
                outline.Append(x, y)
            board.Add(zone)
            zones += 1
board.BuildConnectivity()
board.Save({str(board_path)!r})
with open({str(summary_path)!r}, "w", encoding="utf-8") as out:
    json.dump({{"pads": pads, "zones": zones, "authorized_thermal_pads": authorized}}, out)
"""
    run_pcbnew_script(script)
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    finally:
        summary_path.unlink(missing_ok=True)


def _stamp_local_ground_keepouts(
    board_path: Path, config: dict[str, Any], zone_name: str,
    *, router_clearance_mm: float, prepare_thermal: bool = True,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Keep parent interconnects out of contract-owned ground fill."""
    from kicraft.autoplacer.routing_board import run_pcbnew_script

    keys = (
        "local_bypasses", "thermal_requirements", "fab_capability",
        "gnd_zone_net", "gnd_zone_margin_mm", "zone_clearance_mm",
        "zone_min_thickness_mm", "gnd_plane_pad_connection",
        "zone_thermal_gap_mm", "zone_thermal_spoke_mm",
        "via_size_mm", "via_drill_mm", "thermal_via_pitch_mm",
        "thermal_pad_area_mm2", "thermal_via_inset_mm", "gnd_escape_length_mm",
        "gnd_zone_layer", "hole_to_hole_min_mm", "hole_clearance_min_mm",
        "escape_via_size_mm", "escape_via_drill_mm",
    )
    cfg = {key: config[key] for key in keys if key in config}
    summary_path = board_path.with_suffix(".ground-reservations.json")
    script = f"""
import json
from pathlib import Path
import pcbnew
import sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})
from kicraft.autoplacer.brain.gnd_pour import pour_gnd_planes, add_gnd_pour_and_thermal_vias
from kicraft.autoplacer.hardware.ground_reservations import reserve_connected_ground, reserve_local_ground_planes
from kicraft.autoplacer.fab_profile import fab_floors
cfg = json.loads({json.dumps(cfg)!r})
path = {str(board_path)!r}
sidecars = [Path(path).with_suffix(suffix) for suffix in (".kicad_pro", ".kicad_dru")]
rules = [(sidecar, sidecar.read_bytes() if sidecar.is_file() else None) for sidecar in sidecars]
def restore_rules():
    for sidecar, content in rules:
        if content is None:
            sidecar.unlink(missing_ok=True)
        else:
            sidecar.write_bytes(content)
nets = {{spec["ground_return"]["net"] for spec in cfg.get("local_bypasses", []) if "ground_return" in spec}}
if {prepare_thermal!r} and cfg.get("thermal_requirements"):
    nets.add(cfg.get("gnd_zone_net", "GND"))
for net in sorted(nets):
    if {prepare_thermal!r} and cfg.get("thermal_requirements") and net == cfg.get("gnd_zone_net", "GND"):
        continue  # The thermal finisher owns both faces for this net.
    pour_gnd_planes(path, {{**cfg, "gnd_zone_net": net}}, layers=("F.Cu", "B.Cu"))
    restore_rules()
preparation = None
if {prepare_thermal!r} and cfg.get("thermal_requirements"):
    try:
        preparation = add_gnd_pour_and_thermal_vias(path, cfg)
    finally:
        restore_rules()
    if not preparation.get("zone_filled") or preparation.get("error") or preparation.get("escape_error"):
        raise ValueError(f"thermal ground preparation failed: {{preparation}}")
board = pcbnew.LoadBoard(path)
result = reserve_local_ground_planes(
    board, cfg.get("local_bypasses", []), {zone_name!r},
    router_clearance_mm={router_clearance_mm!r},
)
if {prepare_thermal!r}:
    result["connected_ground"] = [
        reserve_connected_ground(
            board, net, {zone_name!r}, width_mm=fab_floors(cfg)["track_mm"],
            router_clearance_mm={router_clearance_mm!r},
        )
        for net in sorted(nets)
    ]
if preparation is not None:
    result["ground_preparation"] = preparation
board.Save(path)
restore_rules()
with open({str(summary_path)!r}, "w") as out:
    json.dump(result, out)
"""
    sidecars = [board_path.with_suffix(suffix) for suffix in (".kicad_pro", ".kicad_dru")]
    rules = [(path, path.read_bytes() if path.is_file() else None) for path in sidecars]
    try:
        run_pcbnew_script(script, deadline=deadline)
        return json.loads(summary_path.read_text(encoding="utf-8"))
    finally:
        summary_path.unlink(missing_ok=True)
        for path, content in rules:
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)


def _strip_router_keepouts(board_path: Path, zone_name: str) -> None:
    """Remove only our input-only areas, retaining authored rule areas."""
    from kicraft.autoplacer.routing_board import run_pcbnew_script

    script = f"""
import pcbnew
board = pcbnew.LoadBoard({str(board_path)!r})
removed = 0
for zone in list(board.Zones()):
    if zone.GetIsRuleArea() and zone.GetZoneName() == {zone_name!r}:
        board.Delete(zone)
        removed += 1
if removed:
    board.BuildConnectivity()
    board.Save({str(board_path)!r})
"""
    sidecars = [board_path.with_suffix(suffix) for suffix in (".kicad_pro", ".kicad_dru")]
    rules = [(path, path.read_bytes() if path.is_file() else None) for path in sidecars]
    try:
        run_pcbnew_script(script)
    finally:
        for path, content in rules:
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)


def _reject_drills_in_assembly_openings(board_path: Path) -> None:
    """Reject via and footprint drills through ordinary assembly openings."""
    from kicraft.autoplacer.routing_board import run_pcbnew_script

    script = f"""
import pcbnew
import sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})
from kicraft.autoplacer.brain.gnd_pour import _authorized_thermal_pad
board = pcbnew.LoadBoard({str(board_path)!r})
violations = []
for item in board.GetTracks():
    if not isinstance(item, pcbnew.PCB_VIA):
        continue
    radius = (item.GetDrillValue() + 1) // 2
    for fp in board.Footprints():
        for pad in fp.Pads():
            if pad.GetAttribute() not in (
                pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN, pcbnew.PAD_ATTRIB_PTH
            ):
                continue
            if pad.GetNetCode() == item.GetNetCode() and _authorized_thermal_pad(fp, pad):
                continue
            expansions = [
                pad.GetSolderMaskExpansion(layer)
                for layer in (pcbnew.F_Mask, pcbnew.B_Mask)
                if pad.IsOnLayer(layer)
            ]
            has_paste = any(pad.IsOnLayer(layer) for layer in (pcbnew.F_Paste, pcbnew.B_Paste))
            if (expansions or has_paste) and pad.HitTest(
                item.GetPosition(), radius + max([0] + expansions)
            ):
                violations.append(fp.GetReference() + "." + pad.GetNumber())
for hole_fp in board.Footprints():
    for hole in hole_fp.Pads():
        drill = hole.GetDrillSize()
        if drill.x <= 0 and drill.y <= 0:
            continue
        # A circumscribing circle is conservative for slotted drills.
        radius = (max(drill.x, drill.y) + 1) // 2
        for fp in board.Footprints():
            for pad in fp.Pads():
                if pad.GetAttribute() not in (pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN):
                    continue
                if (hole.GetNetCode() != 0 and hole.GetNetCode() == pad.GetNetCode()
                        and _authorized_thermal_pad(fp, pad)):
                    continue
                layers = (pcbnew.F_Mask, pcbnew.B_Mask)
                expansions = [pad.GetSolderMaskExpansion(layer) for layer in layers
                              if pad.IsOnLayer(layer)]
                has_paste = any(pad.IsOnLayer(layer) for layer in (pcbnew.F_Paste, pcbnew.B_Paste))
                if (expansions or has_paste) and pad.HitTest(
                    hole.GetPosition(), radius + max([0] + expansions)
                ):
                    violations.append(
                        hole_fp.GetReference() + "." + hole.GetNumber() + " drill -> "
                        + fp.GetReference() + "." + pad.GetNumber()
                    )
if violations:
    raise RuntimeError("Drill intersects assembly mask or paste opening(s): " + ", ".join(violations))
"""
    run_pcbnew_script(script)


def _board_has_assembly_pads(board_path: Path) -> bool:
    """Avoid a native staging pass for boards without assembly pads."""
    text = board_path.read_text(encoding="utf-8", errors="replace")
    return bool(re.search(r'\(pad\s+(?:"[^"]*"|[^\s()]+)\s+(?:smd|connect|thru_hole)\b', text))


def _lock_input_copper_nets(board_path: Path) -> list[str]:
    """Protect each input-copper net from KRT's automatic reconciliation rip.

    KRT treats any locked track or via as protection for its entire net. Only
    temporary locks are recorded, so authored locks survive the route unchanged.
    """
    from kicraft.autoplacer.routing_board import run_pcbnew_script

    manifest = board_path.with_suffix(".temporary-copper-locks.json")
    script = f"""
import json
import pcbnew
board = pcbnew.LoadBoard({str(board_path)!r})
tracks = list(board.GetTracks())
locked_nets = {{t.GetNetCode() for t in tracks if t.IsLocked()}}
temporary = []
for track in tracks:
    net = track.GetNetCode()
    if net <= 0 or net in locked_nets:
        continue
    track.SetLocked(True)
    locked_nets.add(net)
    temporary.append(track.m_Uuid.AsString())
if temporary:
    board.Save({str(board_path)!r})
with open({str(manifest)!r}, "w", encoding="utf-8") as out:
    json.dump(temporary, out)
"""
    try:
        run_pcbnew_script(script)
        return json.loads(manifest.read_text(encoding="utf-8"))
    finally:
        manifest.unlink(missing_ok=True)


def _restore_input_copper_locks(board_path: Path, temporary: list[str]) -> None:
    """Remove only our temporary locks, without changing authored locks."""
    if not temporary:
        return
    from kicraft.autoplacer.routing_board import run_pcbnew_script

    script = f"""
import pcbnew
board = pcbnew.LoadBoard({str(board_path)!r})
temporary = set({temporary!r})
for track in board.GetTracks():
    if track.m_Uuid.AsString() in temporary:
        track.SetLocked(False)
board.Save({str(board_path)!r})
"""
    run_pcbnew_script(script)


def _routing_project(input_board: Path, config: dict[str, Any]) -> Path:
    expected = input_board.with_suffix(".kicad_pro")
    candidates: list[Path] = []
    configured = str(config.get("pcb_path", "") or "").strip()
    if configured:
        candidates.append(Path(os.path.expanduser(configured)).absolute().with_suffix(".kicad_pro"))
    candidates.extend(sorted(input_board.parent.glob("*.kicad_pro")))
    project = expected if expected.is_file() else next((p for p in candidates if p.is_file()), None)
    if project is None:
        raise KicadRoutingToolsUnavailableError(
            "KiCadRoutingTools requires a sibling .kicad_pro before routing "
            f"{input_board}"
        )
    return project


def net_class_geometry(
    board_path: str | Path, config: dict[str, Any], net_names: list[str] | set[str],
) -> dict[str, dict[str, Any]]:
    """Resolve authoritative project net classes using the pinned router.

    Unlike routing width groups, this must not substitute a finer escape
    class or silently use board-embedded/default geometry.
    """
    board = Path(board_path).resolve()
    project = board.with_suffix(".kicad_pro")
    if not project.is_file():
        raise KicadRoutingToolsUnavailableError(
            f"Net-class geometry requires the board's sibling project: {project}"
        )
    # Parse the project before invoking the upstream reader, which otherwise
    # silently falls back to legacy board rules after malformed project JSON.
    body = json.loads(project.read_text(encoding="utf-8"))
    settings = body.get("net_settings", {})
    classes = settings.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"Project has no authoritative net classes: {project}")
    class_names = [item.get("name") if isinstance(item, dict) else None for item in classes]
    if (any(not isinstance(name, str) or not name for name in class_names)
            or len(set(class_names)) != len(class_names)):
        raise ValueError(f"Project has malformed or duplicate net classes: {project}")
    assignments = settings.get("netclass_assignments") or {}
    patterns = settings.get("netclass_patterns") or []
    if not isinstance(assignments, dict) or not isinstance(patterns, list):
        raise ValueError(f"Project has malformed net-class assignments: {project}")
    for name, assigned in assignments.items():
        values = assigned if isinstance(assigned, list) else [assigned]
        if (not isinstance(name, str) or not name or not values
                or any(not isinstance(value, str) or value not in class_names for value in values)):
            raise ValueError(f"Project has malformed net-class assignment for {name!r}")
    for pattern in patterns:
        if (not isinstance(pattern, dict) or not isinstance(pattern.get("pattern"), str)
                or pattern.get("netclass") not in class_names):
            raise ValueError(f"Project has malformed net-class pattern: {pattern!r}")
    floors = _project_routing_floors(project, config)
    if any(not isinstance(name, str) or not name for name in net_names):
        raise ValueError("Net names must be nonempty strings")
    names = sorted(set(net_names))
    if not names:
        return {}
    runtime = preflight_kicad_routing_tools(config)
    script = """
import contextlib, json, math, os, sys
from list_nets import read_design_rules, resolve_net_class
board, names = sys.argv[1], json.loads(sys.argv[2])
with contextlib.redirect_stdout(sys.stderr):
    rules = read_design_rules(board)
    if os.path.realpath(rules.get('source') or '') != os.path.realpath(os.path.splitext(board)[0] + '.kicad_pro'):
        raise ValueError('net classes did not originate in the sibling project')
    result = {}
    for name in names:
        cls = resolve_net_class(name, rules)
        values = rules['classes'][cls]
        row = {'class_name': cls}
        for key in ('track_width', 'via_diameter', 'via_drill'):
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'invalid {key} for {name!r}: {value!r}')
            row[key] = float(value)
        result[name] = row
print(json.dumps(result))
"""
    try:
        result = subprocess.run(
            [runtime["python"], "-c", script, str(board), json.dumps(names)],
            cwd=Path(runtime["root"]) / "py_router",
            capture_output=True, text=True, check=True,
            timeout=float(config.get("kicad_routing_tools_timeout_s", 120)),
        )
    except subprocess.TimeoutExpired as exc:
        raise KicadRoutingToolsTimeoutError("KiCadRoutingTools timed out resolving net classes") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"KiCadRoutingTools could not resolve net classes: {exc.stderr.strip()}") from exc
    geometry = json.loads(result.stdout)
    for name, row in geometry.items():
        width, diameter, drill = (row[key] for key in ("track_width", "via_diameter", "via_drill"))
        if (width < floors["track_width"] or diameter < floors["via_diameter"]
                or drill < floors["via_drill"] or diameter <= drill
                or (diameter - drill) / 2 < floors["annular"] - 1e-9):
            raise ValueError(f"Net class for {name!r} is below project or fabrication floors")
    if set(geometry) != set(names):
        raise ValueError("Incomplete net-class geometry")
    return geometry


def _width_groups(
    board: Path, runtime: dict[str, str], floor: float, timeout: float,
    config: dict[str, Any] | None = None,
) -> list[tuple[float, list[str]]]:
    """Resolve class widths and validate each ground-last CLI selection."""
    config = config or {}
    ground_nets = set()
    zone_net = config.get("gnd_zone_net", "GND")
    if isinstance(zone_net, str) and zone_net:
        ground_nets.add(zone_net)
    for spec in config.get("local_bypasses", []) or []:
        ground = spec.get("ground_return") if isinstance(spec, dict) else None
        if isinstance(ground, dict) and all(
            isinstance(ground.get(key), str) and ground[key].strip()
            for key in ("owner_ref", "owner_pin", "member_pin", "net")
        ):
            ground_nets.add(ground["net"])
    script = """
import contextlib, json, math, sys
from kicad_parser import parse_kicad_pcb
from list_nets import read_design_rules, net_track_width_map_by_id
from net_queries import expand_net_patterns
board, floor, ground_nets = sys.argv[1], float(sys.argv[2]), set(json.loads(sys.argv[3]))
with contextlib.redirect_stdout(sys.stderr):
    pcb = parse_kicad_pcb(board)
    rules = read_design_rules(board)
    default = rules['classes']['Default']['track_width']
    if isinstance(default, bool) or not isinstance(default, (int, float)) or not math.isfinite(default) or default <= 0:
        raise ValueError('invalid Default net-class track width')
    names = {nid: net.name for nid, net in pcb.nets.items() if nid and net.name}
    widths = net_track_width_map_by_id(board, names, rules)
    selected = set(expand_net_patterns(pcb, ['*']))
    groups = {}
    for nid, name in names.items():
        if name not in selected:
            continue
        width = widths.get(nid, default)
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) or width <= 0:
            raise ValueError(f'invalid track width for {name!r}: {width!r}')
        groups.setdefault((name in ground_nets, max(floor, width)), set()).add(name)
    if set().union(*groups.values()) != selected:
        raise ValueError('router selected nets absent from its net-id table')
    for (ground, width), group in groups.items():
        patterns = sorted(group)
        if any(name.startswith('-') for name in patterns):
            raise ValueError('net name cannot be represented in --nets: ' + repr(patterns))
        expanded = set(expand_net_patterns(pcb, patterns))
        if expanded != group:
            raise ValueError(f'ambiguous net selection for ground={ground}, width {width}: intended {sorted(group)!r}, expanded {sorted(expanded)!r}')
    result = [[width, sorted(group)] for (ground, width), group in sorted(groups.items(), key=lambda item: (item[0][0], -item[0][1]))]
    if not result:
        result = [[max(floor, default), ['*']]]
print(json.dumps(result))
"""
    try:
        result = subprocess.run(
            [runtime["python"], "-c", script, str(board), str(floor), json.dumps(sorted(ground_nets))],
            cwd=Path(runtime["root"]) / "py_router",
            capture_output=True, text=True, check=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise KicadRoutingToolsTimeoutError("KiCadRoutingTools timed out resolving net widths") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"KiCadRoutingTools could not resolve width groups: {exc.stderr.strip()}") from exc
    return [(float(width), list(nets)) for width, nets in json.loads(result.stdout)]


def route_with_kicad_routing_tools(
    kicad_pcb_path: str, output_path: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Route widest classes first; planes replace only natively proved grounds."""
    input_board = Path(kicad_pcb_path).resolve()
    output_board = Path(output_path).resolve()
    if input_board == output_board:
        raise ValueError("KiCadRoutingTools input and output board paths must be distinct")
    output_board.unlink(missing_ok=True)
    runtime = preflight_kicad_routing_tools(config)
    project = _routing_project(input_board, config)
    floors = _project_routing_floors(project, config)
    timeout = float(config.get("kicad_routing_tools_timeout_s", 120))
    started = time.monotonic()
    staged = output_board.with_name(f"{output_board.stem}.krt-width-input.kicad_pcb")
    shutil.copy2(input_board, staged)
    _propagate_sibling_project_rules(str(project.with_suffix(".kicad_pcb")), str(staged))
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        raise KicadRoutingToolsTimeoutError(
            f"KiCadRoutingTools timed out after {timeout:g}s before resolving net widths; "
            f"staged input retained at {staged}"
        )
    groups = _width_groups(staged, runtime, floors["track_width"], remaining, config)
    ground_keepouts = None
    ground_zone_name = f"KiCraft local ground reservation {uuid.uuid4().hex}"
    if config.get("routing_local_ground_reservations"):
        try:
            ground_keepouts = _stamp_local_ground_keepouts(
                staged, config, ground_zone_name,
                router_clearance_mm=floors["clearance"],
                deadline=started + timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise KicadRoutingToolsTimeoutError(
                f"KiCadRoutingTools timed out after {timeout:g}s during ground preparation; "
                f"staged input retained at {staged}"
            ) from exc
        _propagate_sibling_project_rules(str(project.with_suffix(".kicad_pcb")), str(staged))
    ground_plan = (
        {"board_path": str(staged), "zone_name": ground_zone_name,
         "sha256": hashlib.sha256(staged.read_bytes()).hexdigest()}
        if ground_keepouts is not None else None
    )
    # Only a successfully constructed complete native forest may replace
    # explicit ground routing. A configured name alone is not a plane proof.
    plane_grounds = {
        row["net"] for row in (ground_keepouts or {}).get("connected_ground", [])
    }
    groups = [
        (width, remaining_nets)
        for width, nets in groups
        if (remaining_nets := [net for net in nets if net not in plane_grounds])
    ]
    passes = []
    current = staged
    if not groups:
        # Keep the hash-bound staged plan immutable even without router passes.
        current = output_board.with_name(f"{output_board.stem}.krt-plane-connected.kicad_pcb")
        shutil.copy2(staged, current)
        _propagate_sibling_project_rules(str(staged), str(current))
    for index, (width, nets) in enumerate(groups, 1):
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise KicadRoutingToolsTimeoutError(
                f"KiCadRoutingTools timed out after {timeout:g}s before pass {index}; "
                f"completed passes retained at {current}"
            )
        candidate = output_board.with_name(f"{output_board.stem}.krt-pass-{index:02d}.kicad_pcb")
        pass_config = dict(config, kicad_routing_tools_timeout_s=remaining, _krt_deadline=started + timeout)
        stats = _route_krt_pass(str(current), str(candidate), pass_config, nets=nets, track_width=width)
        passes.append(stats)
        current = candidate
    if ground_keepouts is not None:
        _strip_router_keepouts(current, ground_zone_name)
        _propagate_sibling_project_rules(str(project.with_suffix(".kicad_pcb")), str(current))
    if time.monotonic() - started >= timeout:
        raise KicadRoutingToolsTimeoutError(
            f"KiCadRoutingTools timed out after {timeout:g}s; routed candidate retained at {current}"
        )
    shutil.copy2(current, output_board)
    _propagate_sibling_project_rules(str(current), str(output_board))
    if time.monotonic() - started >= timeout:
        output_board.unlink(missing_ok=True)
        raise KicadRoutingToolsTimeoutError(
            f"KiCadRoutingTools timed out after {timeout:g}s; routed candidate retained at {current}"
        )
    if passes:
        last = dict(passes[-1])
    else:
        from .routing_board import import_routed_copper
        from .brain.copper_accounting import fingerprint_trace, fingerprint_via

        input_copper = import_routed_copper(str(input_board))
        output_copper = import_routed_copper(str(output_board))
        preservation = {
            kind: _preservation_group(input_copper.get(kind, []), output_copper.get(kind, []), fingerprint)
            for kind, fingerprint in (("traces", fingerprint_trace), ("vias", fingerprint_via))
        }
        preserved = all(not group["missing_count"] for group in preservation.values())
        last = {
            "backend": "kicad-routing-tools", "version": runtime["version"],
            "commit": runtime["commit"], "source_version": runtime["version"],
            "source_commit": runtime["commit"], "native_version": runtime["native_version"],
            "command": None, "returncode": None, "external_router_run": False,
            "input_copper_preservation": preservation, "preserved_existing_copper": preserved,
            "fabrication_floors": floors,
        }
        if not preserved:
            raise RoutingCopperPreservationError("plane preparation failed to preserve input copper", last)
    def total(key: str) -> float | None:
        values = [item[key] for item in passes]
        return sum(values) if all(isinstance(v, (int, float)) for v in values) else None
    last.update(
        source_input_path=str(input_board),
        local_ground_reservations=ground_keepouts,
        plane_connected_ground_nets=sorted(plane_grounds),
        elapsed_s=round(time.monotonic() - started, 3),
        successful_nets=total("successful_nets"),
        failed_nets=total("failed_nets"),
        total_vias=total("total_vias"),
        router_time_s=total("router_time_s"),
        json_summaries=[summary for item in passes for summary in item["json_summaries"]],
        passes=passes,
    )
    if ground_plan is not None:
        last["local_ground_reservation_plan"] = ground_plan
    return last


def _route_krt_pass(
    kicad_pcb_path: str,
    output_path: str,
    config: dict[str, Any],
    *, nets: list[str], track_width: float,
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

    source_project = _routing_project(input_board, config)

    floors = _project_routing_floors(source_project, config)
    router_input, npth_keepouts = _stage_npth_keepouts(
        input_board,
        output_board,
        source_project,
        hole_clearance_mm=_project_min_hole_clearance(source_project),
        router_clearance_mm=floors["clearance"],
    )
    pad_via_keepouts = None
    pad_via_zone_name = f"{_PAD_VIA_KEEPOUT_ZONE_NAME} {uuid.uuid4().hex}"
    if _board_has_assembly_pads(router_input):
        _reject_drills_in_assembly_openings(router_input)
        pad_via_keepouts = _stamp_pad_via_keepouts(router_input, pad_via_zone_name)
        _propagate_sibling_project_rules(
            str(source_project.with_suffix(".kicad_pcb")), str(router_input)
        )
    temporary_locks = _lock_input_copper_nets(router_input)
    if temporary_locks:
        _propagate_sibling_project_rules(
            str(source_project.with_suffix(".kicad_pcb")), str(router_input)
        )


    fab_overrides = output_board.with_suffix(".fab-overrides.txt")
    fab_overrides.write_text(
        "".join(f"{key} = {value:.9g}\n" for key, value in floors.items()),
        encoding="utf-8",
    )
    command = _krt_command(
        str(router_input), str(output_board), config, fab_overrides,
        nets=nets, track_width=track_width,
        # KRT's pad-centre swap/rescue bypasses via-only rule areas unless its
        # same-net pad policy is enabled. Existing reviewed VIPs remain input
        # copper; new thermal VIPs belong to the qualified ground finisher.
        same_net_pad_clearance=floors["clearance"] if pad_via_keepouts else None,
    )
    timeout_s = float(config.get("kicad_routing_tools_timeout_s", 120))
    if "_krt_deadline" in config:
        timeout_s = min(timeout_s, config["_krt_deadline"] - time.monotonic())
    if timeout_s <= 0:
        raise KicadRoutingToolsTimeoutError(
            f"KiCadRoutingTools timed out before routing; staged input retained at {router_input}"
        )
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
        if not timed_out and proc.returncode == 0:
            _restore_input_copper_locks(output_board, temporary_locks)
        _propagate_sibling_project_rules(str(router_input), str(output_board))
        if npth_keepouts is not None:
            # The router consumed the keepouts; the routed copper already meets
            # the project rule, so the shipped board keeps only authored geometry.
            _strip_npth_keepouts(output_board)
            _propagate_sibling_project_rules(str(router_input), str(output_board))
        if pad_via_keepouts is not None:
            _strip_router_keepouts(output_board, pad_via_zone_name)
            _propagate_sibling_project_rules(str(router_input), str(output_board))
    elapsed = time.monotonic() - started
    stdout_path = output_board.with_suffix(".router.stdout.log")
    stderr_path = output_board.with_suffix(".router.stderr.log")
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    evidence = f"stdout={stdout_path}; stderr={stderr_path}; rules={fab_overrides}"
    if timed_out:
        raise KicadRoutingToolsTimeoutError(
            f"KiCadRoutingTools timed out after {timeout_s:g}s; {evidence}"
        )
    if proc.returncode != 0 or not output_board.is_file():
        detail = (stderr or stdout or "no output").strip()[-4000:]
        raise RuntimeError(f"KiCadRoutingTools failed (rc={proc.returncode}): {detail}; {evidence}")
    if not output_board.with_suffix(".kicad_pro").is_file():
        raise KicadRoutingToolsUnavailableError(
            "KiCadRoutingTools requires a sibling .kicad_pro on its routed output; "
            f"could not propagate project rules to {output_board}"
        )

    if pad_via_keepouts is not None:
        _reject_drills_in_assembly_openings(output_board)

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
        "assembly_pad_via_keepouts": pad_via_keepouts,
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
        missing_traces = int(preservation["traces"]["missing_count"])
        missing_vias = int(preservation["vias"]["missing_count"])
        declared = _declared_input_segment_removals(stdout)
        # The router uses the same "dead-end" wording for copper stripped
        # after ripping and rerouting a whole net. Its count is not proof
        # that the missing input was unloaded; do not waive custody on it.
        raise RoutingCopperPreservationError(
            "KiCadRoutingTools failed to preserve input copper "
            f"(missing traces={missing_traces}, vias={missing_vias}; "
            f"declared dead-end removals={declared}); "
            f"routed output retained at {output_board}",
            stats,
        )
    return stats
