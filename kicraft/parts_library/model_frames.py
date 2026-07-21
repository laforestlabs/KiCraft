"""Do a bundle's two 3D models (.wrl and .step) sit in the same native frame?

A part bundle vendors the same solid twice: a VRML ``.wrl`` (colored, used by
the 3D viewer and renders) and a ``.step`` (used by the fab STEP export). The
footprint's single ``(model ...)`` offset/rotate stanza is applied to BOTH at
export time -- ``kicad-cli pcb export step --subst-models`` swaps the ``.wrl``
reference for the same-named ``.step`` -- so the two files must be authored in
the SAME native frame. easyeda2kicad bakes its placement transform into the
WRL's vertex coordinates but writes the STEP exactly as downloaded from LCSC
(a SolidWorks export in whatever frame the vendor drew), so a freshly fetched
bundle is frequently translation-shifted between the two: the fab STEP then
embeds that part's solid detached from the board (2026-07-21 sweep: 31 of 93
vendored bundles, offsets up to 49.5 mm). ``scripts/restep_model_frames.py``
re-frames a mismatched ``.step``; validate-part check (10) calls
:func:`frame_registry_error` to keep bundles honest.

Geometry sources: the WRL tessellation's ``point [...]`` arrays (VRML unit =
0.1 inch -> mm via x2.54) give a dense cloud of the whole surface; the STEP's
topological vertices (3-coordinate ``CARTESIAN_POINT`` entities referenced by
``VERTEX_POINT``) give a sparse-but-exact skeleton. When frames agree, every
STEP vertex lies on (or within tessellation error of) the WRL surface, so the
median nearest-neighbor distance is a robust registry error: ~0.0-0.4 mm for
matched frames, the full offset magnitude for mismatched ones. The comparison
is deliberately one-directional -- the WRL may legitimately contain MORE
geometry than the STEP (as312's dome), but every STEP vertex must have WRL
support.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

WRL_UNIT_MM = 2.54  # 1 VRML unit = 0.1 inch

_NUM = r"(-?[\d.]+(?:[eE][+-]?\d+)?)"
_WRL_TRIPLE = re.compile(_NUM + r"\s+" + _NUM + r"\s+" + _NUM)
_WRL_POINT_BLOCK = re.compile(r"point\s*\[(.*?)\]", re.S)
_STEP_CARTESIAN = re.compile(
    r"#(\d+)\s*=\s*CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(([^)]*)\)\s*\)"
)
_STEP_VERTEX = re.compile(r"VERTEX_POINT\s*\(\s*'[^']*'\s*,\s*#(\d+)")


def wrl_points(path: Path, limit: int = 8000) -> np.ndarray:
    """Surface point cloud of a VRML model, in mm, deterministically thinned."""
    txt = Path(path).read_text(errors="replace")
    pts: list[tuple[float, float, float]] = []
    for block in _WRL_POINT_BLOCK.findall(txt):
        for t in _WRL_TRIPLE.findall(block):
            pts.append((float(t[0]), float(t[1]), float(t[2])))
    arr = np.asarray(pts, dtype=float) * WRL_UNIT_MM
    if len(arr) > limit:
        arr = arr[:: len(arr) // limit + 1]
    return arr


def step_vertex_points(path: Path) -> np.ndarray:
    """Topological vertices of a STEP model (mm): 3-coordinate
    CARTESIAN_POINTs referenced by VERTEX_POINT entities."""
    txt = Path(path).read_text(encoding="latin-1")
    cart: dict[int, tuple[float, ...]] = {}
    for m in _STEP_CARTESIAN.finditer(txt):
        coords = tuple(float(v) for v in m.group(2).split(","))
        if len(coords) == 3:
            cart[int(m.group(1))] = coords
    verts = [cart[int(i)] for i in _STEP_VERTEX.findall(txt) if int(i) in cart]
    return np.asarray(verts, dtype=float)


def nearest_distances(src: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """For each src point, distance to its nearest cloud point (chunked)."""
    out = np.empty(len(src))
    for i in range(0, len(src), 256):
        chunk = src[i : i + 256]
        d = np.linalg.norm(chunk[:, None, :] - cloud[None, :, :], axis=2)
        out[i : i + 256] = d.min(axis=1)
    return out


def frame_registry_error(wrl_path: Path, step_path: Path) -> float | None:
    """Lower-quartile distance from the STEP's vertices to the WRL cloud, mm.

    The lower quartile (not the median) is deliberate: the vendor's two
    solids often differ in artwork detail (lead lengths, body domes), which
    inflates the median even when the frames coincide -- but in a matched
    frame a large fraction of STEP vertices sit exactly on WRL points, so
    the quartile stays ~0.0-0.1 (worst observed on a matched bundle: 0.08).
    A rigid offset moves every vertex, so the quartile reads the offset.
    None when either file yields no usable geometry (nothing to compare --
    callers should treat that as a pass).
    """
    cloud = wrl_points(wrl_path)
    verts = step_vertex_points(step_path)
    if len(cloud) < 10 or len(verts) < 4:
        return None
    return float(np.percentile(nearest_distances(verts, cloud), 25))


_Z_ROTATIONS = {
    0: np.eye(3),
    90: np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    180: np.array([[-1.0, 0, 0], [0, -1, 0], [0, 0, 1]]),
    270: np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
}
_ASSEMBLY_ENTITIES = re.compile(
    r"MAPPED_ITEM|ITEM_DEFINED_TRANSFORMATION|CARTESIAN_TRANSFORMATION_OPERATOR"
    r"|NEXT_ASSEMBLY_USAGE"
)
_POINT_ENTITY = re.compile(r"CARTESIAN_POINT\s*\(\s*('[^']*')\s*,\s*\(([^)]*)\)\s*\)")
_DIRECTION_ENTITY = re.compile(r"DIRECTION\s*\(\s*('[^']*')\s*,\s*\(([^)]*)\)\s*\)")


def _fmt_coord(v: float) -> str:
    if abs(v) < 5e-13:
        v = 0.0
    return f"{v:.12g}"


def transform_step_file(
    path: Path, deg: int, t: tuple[float, float, float]
) -> tuple[int, int]:
    """Rigidly transform a STEP file in place: every 3-coordinate
    CARTESIAN_POINT becomes R.p + t and (when deg != 0) every 3-coordinate
    DIRECTION becomes R.d, where R is the z-rotation by ``deg`` degrees.
    Returns (#points rewritten, #directions rewritten). Refuses files with
    assembly transforms -- a flat rewrite would double-apply there."""
    path = Path(path)
    txt = path.read_text(encoding="latin-1")
    if _ASSEMBLY_ENTITIES.search(txt):
        raise RuntimeError(
            f"{path.name}: contains assembly transforms; a flat point rewrite "
            f"would double-apply -- handle manually"
        )
    rot = _Z_ROTATIONS[deg]
    tv = np.asarray(t, dtype=float)
    counts = [0, 0]

    def sub_point(m: re.Match) -> str:
        coords = [float(v) for v in m.group(2).split(",")]
        if len(coords) != 3:
            return m.group(0)
        counts[0] += 1
        p = rot @ np.asarray(coords) + tv
        return (
            f"CARTESIAN_POINT ( {m.group(1)},  "
            f"( {_fmt_coord(p[0])}, {_fmt_coord(p[1])}, {_fmt_coord(p[2])} ) )"
        )

    def sub_dir(m: re.Match) -> str:
        coords = [float(v) for v in m.group(2).split(",")]
        if len(coords) != 3:
            return m.group(0)
        counts[1] += 1
        d = rot @ np.asarray(coords)
        return (
            f"DIRECTION ( {m.group(1)}, "
            f"( {_fmt_coord(d[0])}, {_fmt_coord(d[1])}, {_fmt_coord(d[2])} ) )"
        )

    txt = _POINT_ENTITY.sub(sub_point, txt)
    if deg != 0:
        txt = _DIRECTION_ENTITY.sub(sub_dir, txt)
    path.write_text(txt, encoding="latin-1")
    return counts[0], counts[1]


def cloud_spacing(cloud: np.ndarray, sample: int = 400) -> float:
    """p90 nearest-neighbor spacing within a point cloud (tessellation pitch)."""
    idx = np.arange(0, len(cloud), max(1, len(cloud) // sample))[:sample]
    picks = cloud[idx]
    dists = np.empty(len(picks))
    for i, p in enumerate(picks):
        d = np.linalg.norm(cloud - p, axis=1)
        d[d == 0.0] = np.inf
        dists[i] = d.min()
    return float(np.percentile(dists, 90))


def frame_mismatch(wrl_path: Path, step_path: Path) -> str | None:
    """Reason string when the .step's native frame disagrees with the .wrl's.

    Two complementary detectors, either one trips:

    * bbox-center delta while the per-axis bbox *sizes* agree (< 1 mm): a
      rigid translation shifts the center and keeps the sizes. This prong
      also catches pitch-periodic shifts (a header strip offset by one pin
      pitch lands most vertices on the neighboring pin's geometry, blinding
      the registry prong). When the sizes disagree, the two files genuinely
      draw different solids and center comparison is meaningless (as312's
      WRL has a dome the STEP lacks) -- then only the registry prong runs.
    * displaced-copy registry: when the identity registry error exceeds the
      WRL's own tessellation pitch (floor 0.5 mm) AND some rigid candidate
      (z-rotation x bbox-centering) restores the registry to under half the
      identity error, the .step is a misplaced copy of the same artwork
      (catches rotations the bbox prong is blind to). A candidate only
      counts if it also keeps the low-z band (pins/leads -- the part of the
      model that must meet the pads) registered: on rotation-symmetric
      bodies a spurious rotation can improve the body score while swinging
      the pins off their pads (pe225j2a). When NO qualifying transform
      helps, the two files simply draw different solids (mes104j2a's coil
      winding vs plain cylinder) -- a modeling difference, not a frame
      defect, so it passes.

    Returns None when the frames agree or there is nothing to compare.
    """
    cloud = wrl_points(wrl_path)
    verts = step_vertex_points(step_path)
    if len(cloud) < 10 or len(verts) < 4:
        return None
    wmin, wmax = cloud.min(axis=0), cloud.max(axis=0)
    smin, smax = verts.min(axis=0), verts.max(axis=0)
    if np.all(np.abs((wmax - wmin) - (smax - smin)) < 1.0):
        delta = (wmin + wmax) / 2 - (smin + smax) / 2
        if np.any(np.abs(delta) > 0.5):
            return (
                f"model frames disagree: .step is translated "
                f"({delta[0]:+.2f}, {delta[1]:+.2f}, {delta[2]:+.2f}) mm "
                f"from the .wrl frame"
            )
    err_id = float(np.percentile(nearest_distances(verts, cloud), 25))
    if err_id <= max(0.5, cloud_spacing(cloud)):
        return None

    def low_band(pts: np.ndarray) -> np.ndarray:
        zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
        return pts[pts[:, 2] < zmin + 0.15 * (zmax - zmin) + 1e-6]

    wband = low_band(cloud)
    pin_id = float(np.percentile(nearest_distances(low_band(verts), wband), 25))
    wctr = (wmin + wmax) / 2
    rots = (
        np.eye(3),
        np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        np.array([[-1.0, 0, 0], [0, -1, 0], [0, 0, 1]]),
        np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    )
    best = err_id
    for rot in rots:
        rp = verts @ rot.T
        rp += wctr - (rp.min(axis=0) + rp.max(axis=0)) / 2
        pin = float(np.percentile(nearest_distances(low_band(rp), wband), 25))
        if pin > max(0.6, 0.5 * pin_id):
            continue  # pins swing off their pads: not a plausible re-frame
        best = min(best, float(np.percentile(nearest_distances(rp, cloud), 25)))
    if best < 0.5 * err_id:
        return (
            f"model frames disagree: .step registers {err_id:.2f} mm off the "
            f".wrl as stored, but {best:.2f} mm after a rigid re-frame -- it "
            f"is a displaced copy of the same solid"
        )
    return None
