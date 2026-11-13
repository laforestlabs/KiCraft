"""The fab capability profile: one place that says what the fab can actually do.

KiCraft's minimum feature sizes were scattered as bare 0.153 constants across
``autoplacer/config.py``, ``design/synthesis/kicad_pro.py`` and the breakout
stamper, all of them encoding the **OSH Park 6 mil** floor (0.1524 mm, raised to
0.153 to clear the router exchange input's whole-micron rounding). The pipeline's actual fab
target is JLCPCB -- the BOM gates on JLC assembly plus LCSC retail -- and JLC's
published 2-layer 1 oz capability is finer in every dimension that matters to a
fine-pitch escape. The legacy floor is what closes the designed escape lanes of
a ring package: lane sharing misses by 15 um, the same-row diagonal by 2 um, and
the 0.6 mm netclass via has no legal position anywhere near an inner ring, so
the dog-bone fanout those packages are *designed* for cannot be built at all.

This module holds the profile and the two accessors everything else reads, so a
capability question has exactly one answer and one place to change it.

Verified against JLCPCB's published capability page (2026-07-23):

* minimum track width / spacing, 2-layer 1 oz: **0.10 mm / 0.10 mm** (4/4 mil)
* minimum via hole: **0.15 mm**; minimum via diameter: **0.25 mm**

We deliberately floor at **0.127 mm (5 mil)**, one step *above* JLC's stated
minimum, so a board that lands exactly on the floor still has fab margin -- and
because 0.127 mm is exactly 127 um, so the router exchange input's integer-micron rounding trap
that motivated 0.153 does not exist there.

The fanout via is **0.36 / 0.15 mm**. Two constraints pin it, pulling opposite
ways, and the size is the balance point between them.

*Upper bound -- it has to fit.* The worst inner-ring pad we have is nRF52840
aQFN-73 AC13: a 0.25 mm channel between the inner ring and a fully populated
outer column, with the exposed pad on the other side. Its legal via-centre
window, against every rule that really applies (netclass clearance 0.153, the
stamper's +10 um geometry guard, KiCad's 0.25 mm hole-to-copper minimum):

    via 0.6/0.3    none      <- what shipped: no legal position at any offset
    via 0.5/0.3    none
    via 0.45/0.2   none
    via 0.4/0.2    none          (17 um at the bare rule; the guard closes it)
    via 0.39/0.15   6.5 um
    via 0.375/0.15 22 um
    via 0.36/0.15  37.6 um   <- chosen
    via 0.35/0.15  48 um

*Lower bound -- the ANNULAR RING has to be wide enough.* KiCad Routing Tools only knows
the copper clearance; it has never heard of hole-to-copper. So a track it places
at exactly the copper rule from a via's annulus is only ``clearance + ring`` from
that via's HOLE -- and if the ring is thin, that is under the hole rule and KiCad
fails the board. The invariant is therefore

    netclass_clearance + annular_ring >= hole_to_copper_clearance

which for 0.153 and 0.25 needs a ring of at least 0.097 mm. A 0.35/0.2 via
(ring 0.075) misses it by 22 um -- and did, immediately: on the witness board
KiCad Routing Tools ran a B.Cu track 0.2417 mm from the nRESET fanout hole against the
0.25 mm rule, and an otherwise perfect zero-unconnected leaf round was thrown
away for it. 0.36/0.15 gives a 0.105 mm ring: +8 um nominal, +18 um once
KiCad Routing Tools's own clearance guard is counted. (Both bounds together force a
drill of 0.18 mm or finer, which is why this class uses 0.15 -- JLC's stated
2-layer minimum -- rather than a more comfortable 0.2.)

Every dimension is inside JLC's published 2-layer capability: 0.36 > 0.25 mm
minimum diameter, 0.15 = minimum hole, 0.105 > 0.05 mm minimum ring. JLC
surcharges any via under 0.45 mm diameter, which is a price line rather than a
capability limit, and it applies only to the handful of fanout vias a trapped
pad needs.

For the default 1 oz process, Default/Power netclasses retain their existing
0.153 mm clearance and the 0.36/0.15 mm fanout via. Recipe-required 2 oz boards
instead enforce at least 0.17 mm track and clearance in all project netclasses
and fabrication floors. The via dimensions do not change.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

__all__ = [
    "FAB_CAPABILITY",
    "NETCLASS_CLEARANCE_MM",
    "fab_floors",
    "fanout_via",
    "stamp_fab_floors_into_pro",
    "stamp_fab_floors_for_project",
]

# The Default/Power netclass clearance generated boards carry. NOT a floor --
# it is what KiCad DRC actually resolves a pair to, and therefore what anything
# planning copper has to hold. Deliberately unchanged by the capability work
# (scope guard): only the floors and the stamped escape copper moved.
NETCLASS_CLEARANCE_MM = 0.153

# JLCPCB's published 2 oz FR4 minimum is 0.16 mm; 0.17 mm leaves 10 µm
# process margin and is exact in whole-micron router exchange files.
# https://jlcpcb.com/help/article/jlcpcb-copper-weight (2026-09-09).
TWO_OUNCE_FEATURE_MM = 0.17
TWO_OUNCE_PTH_ANNULUS_MM = 0.254
TWO_OUNCE_HOLE_GAP_MM = 0.2


def requires_two_ounce(cfg: dict[str, Any] | None) -> bool:
    """Whether the selected, recipe-owned thermal contract requires 2 oz."""
    return any(float(row.get("outer_copper_oz", 0)) >= 2.0
               for row in ((cfg or {}).get("thermal_requirements") or []))


# JLCPCB 2-layer, 1 oz, standard process. See the module docstring for the
# provenance of each number and why the track/clearance floor sits above the
# published minimum.
FAB_CAPABILITY: dict[str, float] = {
    "min_track_mm": 0.127,
    "min_clearance_mm": 0.127,
    # The fanout-via class -- NOT the netclass via (which stays 0.6/0.3).
    # Sized by two opposing constraints; see the module docstring. Changing
    # either number without re-deriving both is a board-breaking edit.
    "min_via_diameter_mm": 0.36,
    "min_via_drill_mm": 0.15,
}


def fab_floors(cfg: dict[str, Any] | None = None) -> dict[str, float]:
    """``{track_mm, clearance_mm}`` -- the finest features the fab will build.

    A project config may override via a ``fab_capability`` block; missing keys
    fall back to :data:`FAB_CAPABILITY`.
    """
    block = dict(FAB_CAPABILITY)
    if cfg:
        override = cfg.get("fab_capability") or {}
        if isinstance(override, dict):
            block.update({k: float(v) for k, v in override.items() if v is not None})
    track = float(block["min_track_mm"])
    clearance = float(block["min_clearance_mm"])
    if requires_two_ounce(cfg):
        if not math.isfinite(track) or not math.isfinite(clearance):
            raise ValueError("invalid 2 oz fabrication floor")
        track = max(track, TWO_OUNCE_FEATURE_MM)
        clearance = max(clearance, TWO_OUNCE_FEATURE_MM)
    return {"track_mm": track, "clearance_mm": clearance}


def fanout_via(cfg: dict[str, Any] | None = None) -> tuple[float, float]:
    """``(diameter_mm, drill_mm)`` of the escape/dog-bone via class.

    Separate from ``via_size_mm``/``via_drill_mm`` (the 0.6/0.3 netclass via the
    router and the pours use): a 0.6 mm via has no legal position beside a
    0.5 mm-pitch inner ring at any clearance, so the fanout class exists purely
    so trapped pads can be escaped. ``escape_via_size_mm`` /
    ``escape_via_drill_mm`` override it per project.
    """
    block = dict(FAB_CAPABILITY)
    if cfg:
        override = (cfg.get("fab_capability") or {}) if isinstance(
            cfg.get("fab_capability"), dict
        ) else {}
        block.update({k: float(v) for k, v in override.items() if v is not None})
    dia = float((cfg or {}).get("escape_via_size_mm") or block["min_via_diameter_mm"])
    drill = float((cfg or {}).get("escape_via_drill_mm") or block["min_via_drill_mm"])
    return dia, drill


def stamp_fab_floors_into_pro(pro_path: str | Path, cfg: dict[str, Any] | None = None) -> bool:
    """Stamp the project's fabrication rules; enforce 2 oz floors fail-closed.

    The floors that gate a fine-pitch escape live in the *project* file, not the
    board, so a board copied forward from a project synthesised before this
    change still carries ``min_via_diameter: 0.508`` -- which fails every 0.4 mm
    fanout via at DRC even though the fab builds them happily. Stamping here
    means ``kicraft replay`` (which re-uses a frozen seed project) exercises the
    capability change instead of silently testing the old floors.

    For 2 oz, trace and clearance floors and netclasses can only increase.
    Existing 1 oz replay behavior and the independently sized via class remain.
    """
    p = Path(pro_path)
    if not p.is_file():
        if requires_two_ounce(cfg):
            raise ValueError(f"missing required 2 oz project rules: {p}")
        return False
    floors = fab_floors(cfg)
    via_d, via_dr = fanout_via(cfg)
    wanted = {
        "min_track_width": floors["track_mm"],
        "min_clearance": floors["clearance_mm"],
        "min_via_diameter": via_d,
        "min_via_annular_width": round((via_d - via_dr) / 2.0, 4),
        # KiCad's "minimum through hole" gates VIA drills too, and its default
        # 0.3 mm is the old netclass via's drill -- it fails every fanout via
        # by construction. PTH *pad* drills are kept at the fab floor by
        # validate-part/add-part (check 6), which does not go through here.
        "min_through_hole_diameter": via_dr,
    }
    if requires_two_ounce(cfg):
        wanted["min_hole_to_hole"] = TWO_OUNCE_HOLE_GAP_MM
    try:
        body = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        if requires_two_ounce(cfg):
            raise ValueError(f"cannot enforce required 2 oz rules in {p}")
        return False
    settings = body.setdefault("board", {}).setdefault("design_settings", {})
    rules = settings.setdefault("rules", {})
    changed = False
    heavy_copper = requires_two_ounce(cfg)
    if heavy_copper:
        classes = body.get("net_settings", {}).get("classes", [])
        if not classes or not any(c.get("name") == "Default" for c in classes):
            raise ValueError(f"cannot enforce required 2 oz netclasses in {p}")
        for netclass in classes:
            if float(netclass.get("clearance") or 0) < floors["clearance_mm"]:
                netclass["clearance"] = floors["clearance_mm"]
                changed = True
            if float(netclass.get("track_width") or 0) < floors["track_mm"]:
                netclass["track_width"] = floors["track_mm"]
                changed = True
    # Make the fanout class a first-class via of the project. KiCad's Specctra
    # export emits one padstack per entry here and lists them all in the router exchange input's
    # (structure (via ...)) rule, so a stamped dog-bone round-trips through the
    # routed session intact -- and a human opening the board gets it in the via dropdown.
    # It does NOT change what KiCad Routing Tools places: each netclass carries its own
    # (use_via ...) naming the netclass via, which stays 0.6/0.3 (scope guard).
    dims = settings.setdefault("via_dimensions", [])
    if isinstance(dims, list) and not any(
        isinstance(d, dict) and abs(float(d.get("diameter", 0.0)) - via_d) < 1e-9
        for d in dims
    ):
        dims.append({"diameter": via_d, "drill": via_dr})
        changed = True
    for key, value in wanted.items():
        current = rules.get(key)
        if (current is None or
                (heavy_copper and key in ("min_track_width", "min_clearance", "min_hole_to_hole")
                 and float(current) < value - 1e-9) or
                (not (heavy_copper and key in ("min_track_width", "min_clearance", "min_hole_to_hole"))
                 and float(current) > value + 1e-9)):
            rules[key] = value
            changed = True
    if changed:
        try:
            p.write_text(json.dumps(body, indent=2) + "\n")
        except OSError:
            if heavy_copper:
                raise
            return False
    return changed


def require_two_ounce_project_rules(pro_path: str | Path, cfg: dict[str, Any]) -> None:
    """Reject a 2 oz project whose actual native DRC/netclass rules are weaker."""
    if not requires_two_ounce(cfg):
        return
    body = json.loads(Path(pro_path).read_text())
    rules = body["board"]["design_settings"]["rules"]
    classes = body["net_settings"]["classes"]
    floors = fab_floors(cfg)
    if not any(c.get("name") == "Default" for c in classes):
        raise ValueError("2 oz project has no Default netclass")
    for key, floor in (("min_track_width", floors["track_mm"]),
                       ("min_clearance", floors["clearance_mm"]),
                       ("min_hole_to_hole", TWO_OUNCE_HOLE_GAP_MM)):
        value = float(rules[key])
        if not math.isfinite(value) or value < floor:
            raise ValueError(f"2 oz project rule {key} is below {floor:g} mm")
    for netclass in classes:
        for key, floor in (("clearance", floors["clearance_mm"]),
                           ("track_width", floors["track_mm"])):
            value = float(netclass.get(key) or 0)
            if not math.isfinite(value) or value < floor:
                raise ValueError(f"2 oz netclass {netclass.get('name')} {key} is below {floor:g} mm")


def stamp_fab_floors_for_project(
    project_dir: str | Path, cfg: dict[str, Any] | None = None
) -> int:
    """Stamp every ``.kicad_pro`` in *project_dir* with the applicable floors.

    Call once, before anything loads a board: the leaf and parent boards written
    from here on copy the project file forward, so this is the single place the
    capability floors have to reach. Returns how many files changed (0 when the
    project already carries them, which is every freshly synthesised project).
    """
    changed = 0
    for pro in sorted(Path(project_dir).glob("*.kicad_pro")):
        if stamp_fab_floors_into_pro(pro, cfg):
            changed += 1
            print(
                f"  fab floors: updated the DRC minimums in {pro.name} for the "
                "fab capability (autoplacer/fab_profile.py)"
            )
    return changed
