"""Re-frame vendored .step models into their bundle's .wrl frame.

Why: the footprint's single ``(model ...)`` transform is applied to both the
.wrl (renders) and the same-named .step (fab STEP export via ``--subst-models``),
so both files must share one native frame. easyeda2kicad bakes its placement
into the WRL's coordinates but stores the STEP raw as downloaded, so fetched
bundles routinely disagree -- the fab STEP then embeds the part detached from
the board. 2026-07-21 sweep: 31 of 93 vendored bundles, offsets 0.5..49.5 mm
(headers half a strip away, BNC 13 mm, DC barrel jack 11 mm above the board).

What it does: rigidly transforms the .step's geometry (every 3-coordinate
CARTESIAN_POINT becomes R.p + t; for R != I every 3-coordinate DIRECTION
becomes R.d) so its frame coincides with the .wrl's. Safe only because these
files carry no assembly transforms (MAPPED_ITEM / ITEM_DEFINED_TRANSFORMATION
/ CARTESIAN_TRANSFORMATION_OPERATOR) -- the tool refuses files that do.

Modes:
  --fit     print best-fit (rotation, translation) candidates per bundle with
            ambiguity flags; use to derive entries for APPLIED_TRANSFORMS
  --verify  print the frame registry error per bundle (should be < ~0.5 mm)
  --apply   apply APPLIED_TRANSFORMS to the library files in place

The APPLIED_TRANSFORMS table below is the *reviewed* artifact: fits were
scored by nearest-neighbor registry against the WRL cloud and the ambiguous
ones adjudicated feature-by-feature (low-z pin/lead clusters vs the footprint
pads -- NEVER from renders; the model itself can be rotated, KC-DVA3UP).
Notable adjudications:
  * aod4185 / az1084c-3v3: body NN slightly preferred 180 deg (symmetric
    molded prisms), but the lead clusters register exactly under rot 0.
  * as312: bodies genuinely differ (WRL has the taller dome/tab), so the
    translation is pin-anchored, not bbox-anchored.
  * ap63203 / ch32v003j4m6 / ttp223: genuinely 180 deg off (pin-1 features).
After --apply, re-bless each bundle: validate-part <dir> --update-hash.
Sub-0.5 mm z-only offsets (chip seating planes) were fixed along the way.

Run with the project venv:  python scripts/restep_model_frames.py --verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kicraft.parts_library.model_frames import (  # noqa: E402
    frame_mismatch,
    frame_registry_error,
    nearest_distances,
    step_vertex_points,
    transform_step_file,
    wrl_points,
)

LIB = Path(__file__).resolve().parent.parent / "kicraft" / "parts_library"

ROTS = {
    0: np.eye(3),
    90: np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    180: np.array([[-1.0, 0, 0], [0, -1, 0], [0, 0, 1]]),
    270: np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
}

# bundle name -> (rotation deg about +z, translation mm), mapping the vendored
# .step's frame onto the .wrl's. Applied 2026-07-21; kept for provenance and
# for re-use after any future re-fetch reintroduces raw-LCSC frames.
APPLIED_TRANSFORMS: dict[str, tuple[int, tuple[float, float, float]]] = {
    "aod4185": (0, (-0.90, 0.00, 0.05)),
    "ap63203": (180, (0.00, 0.00, 0.62)),
    "as312": (0, (0.00, 1.25, 0.00)),  # pin-anchored; bodies differ
    "az1084c-3v3": (0, (-1.33, -4.94, 0.00)),
    "b2b-xh-a-lf-sn": (0, (-1.25, -1.78, 1.80)),
    "banana-jack-black": (0, (-9.71, 0.00, 6.55)),
    "banana-jack-red": (0, (-9.71, 0.00, 6.55)),
    "bnc-pcb-jack": (0, (0.00, -13.25, 0.60)),
    "ch224k": (0, (0.00, 0.00, 0.81)),
    "ch32v003j4m6": (180, (0.00, 0.00, 0.85)),
    "ch340n": (0, (-0.01, 0.07, 0.80)),
    "dc-barrel-jack-5-5-2-1": (0, (-1.39, 0.00, 11.00)),
    "ds18b20": (0, (0.00, 1.03, 0.00)),
    "esp32-s3-mini-1": (0, (0.00, 2.54, 0.02)),
    "esp32-s3-wroom-1": (0, (0.00, 3.65, 0.01)),
    "esp32-wroom-32e-n4": (0, (-3.78, 0.00, 0.00)),
    "header-male-2-54-1x40": (0, (49.53, 0.00, 0.00)),
    "hs96l03w2c03": (0, (0.00, -12.66, -2.38)),
    "jst-ph-2p": (0, (1.00, -2.75, 0.00)),
    # mes104j2a-7-50r0 deliberately absent: its wrl (winding-coil artwork) and
    # step (plain case cylinder) are different solids; pins register at rot 0
    # and no rigid transform improves the registry -- not a frame defect.
    "jst-sh-4p-qwiic": (0, (-1.50, -0.35, 0.01)),
    "microsd-tf01a": (0, (-0.03, -5.28, 0.15)),
    "mp1584en": (0, (0.00, 0.00, 0.85)),
    "pam8302a": (0, (-0.01, 0.07, 0.80)),
    "pin-header-female-2-54-1x40": (0, (49.53, 0.00, 0.00)),
    "screw-terminal-5mm-2p": (0, (2.39, 0.35, 0.00)),
    "screw-terminal-5mm-3p": (0, (4.99, 0.00, 0.00)),
    "ss14": (0, (0.00, 0.00, 1.17)),
    "ttp223": (180, (0.00, 0.00, 0.75)),
    "usb-micro-b-receptacle-5p": (0, (-0.51, 0.31, -0.10)),
    "veml7700": (0, (0.00, 0.44, 1.49)),
    "vl53l0x": (0, (0.00, 0.00, 1.02)),
}

def bundle_model_pair(part_dir: Path) -> tuple[Path, Path] | None:
    td = part_dir / "3d"
    if not td.is_dir():
        return None
    wrls = sorted(td.glob("*.wrl"))
    steps = sorted(td.glob("*.step")) + sorted(td.glob("*.stp"))
    if not wrls or not steps:
        return None
    return wrls[0], steps[0]


def cmd_verify(bundles: list[Path]) -> int:
    worst = 0.0
    bad = 0
    for d in bundles:
        pair = bundle_model_pair(d)
        if pair is None:
            continue
        err = frame_registry_error(*pair)
        if err is None:
            print(f"{d.name:32s} (no comparable geometry)")
            continue
        worst = max(worst, err)
        reason = frame_mismatch(*pair)
        if reason:
            bad += 1
        print(f"{d.name:32s} registry q25 {err:7.3f} mm"
              + (f"  <-- {reason}" if reason else ""))
    print(f"\nworst q25 {worst:.3f} mm; {bad} mismatched bundle(s)")
    return 1 if bad else 0


def cmd_fit(bundles: list[Path]) -> int:
    for d in bundles:
        pair = bundle_model_pair(d)
        if pair is None:
            continue
        cloud = wrl_points(pair[0])
        verts = step_vertex_points(pair[1])
        if len(cloud) < 10 or len(verts) < 4:
            continue
        wc = (cloud.min(axis=0) + cloud.max(axis=0)) / 2
        scored = []
        for deg, R in ROTS.items():
            rp = verts @ R.T
            t = wc - (rp.min(axis=0) + rp.max(axis=0)) / 2
            med = float(np.median(nearest_distances(rp + t, cloud)))
            scored.append((med, deg, t))
        scored.sort()
        med, deg, t = scored[0]
        flag = "" if med < 0.4 and scored[1][0] > 2 * med else "  <-- AMBIGUOUS, adjudicate by pin features"
        print(
            f"{d.name:32s} rot={deg:3d} t=({t[0]:7.2f},{t[1]:7.2f},{t[2]:6.2f}) "
            f"med={med:6.3f} next={scored[1][0]:6.3f}{flag}"
        )
    return 0


def cmd_apply(bundles: list[Path]) -> int:
    names = {d.name for d in bundles}
    for name, (deg, t) in APPLIED_TRANSFORMS.items():
        if name not in names:
            continue
        pair = bundle_model_pair(LIB / name)
        if pair is None:
            print(f"{name}: no wrl+step pair, skipped", file=sys.stderr)
            continue
        before = frame_registry_error(*pair)
        pts, dirs = transform_step_file(pair[1], deg, t)
        after = frame_registry_error(*pair)
        reason = frame_mismatch(*pair)
        print(
            f"{name:32s} rot={deg:3d} t=({t[0]:7.2f},{t[1]:7.2f},{t[2]:6.2f}) "
            f"points={pts} dirs={dirs}  registry q25 {before:.3f} -> {after:.3f} mm"
        )
        if reason:
            print(f"  WARNING {name}: still mismatched after transform: {reason}", file=sys.stderr)
    print("\nnow re-bless hashes: python -m kicraft.design.cli_app validate-part <dir> --update-hash")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fit", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--apply", action="store_true")
    ap.add_argument("bundles", nargs="*", help="bundle names (default: all)")
    args = ap.parse_args()

    if args.bundles:
        dirs = [LIB / b for b in args.bundles]
    else:
        dirs = sorted(d for d in LIB.iterdir() if d.is_dir() and (d / "3d").is_dir())
    if args.fit:
        return cmd_fit(dirs)
    if args.verify:
        return cmd_verify(dirs)
    return cmd_apply(dirs)


if __name__ == "__main__":
    raise SystemExit(main())
