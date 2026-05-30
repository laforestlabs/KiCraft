"""Library invariant: antenna-bearing module footprints carry an RF keep-out.

The KiCraft parts library vendors RF module footprints (ESP32 WROOM family
today). easyeda2kicad imports routinely *drop* the antenna keep-out zone that
the stock KiCad footprints carry, which lets the placer/router treat the
antenna near-field as free space (parts placed beside the antenna, traces run
under it). Fix 0 repaired the two ESP32 modules; this test makes the invariant
durable so a future re-vendored footprint that regresses the keep-out fails CI
instead of shipping silently.

Policy (matches the on-module strip baked by Fix 0 — the larger RF near-field
clearance is enforced at placement time, not in the .kicad_mod):
  * every antenna-bearing module has >=1 footprint-internal rule-area zone with
    tracks/copperpour/footprints ``not_allowed`` over the antenna, and
  * a non-empty courtyard covering the module body.

Discovery is by footprint-name pattern; a guard test asserts the two known
modules are found so the parametrization can never vacuously pass.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pcbnew = pytest.importorskip("pcbnew")

_PARTS_LIBRARY = Path(__file__).resolve().parents[1] / "kicraft" / "parts_library"

# Antenna-bearing RF module families. Matched against the .kicad_mod stem.
_ANTENNA_NAME_RE = re.compile(r"WROOM|WIFI|BLE|NRF|ESP32|ESP8266", re.IGNORECASE)

# Modules that MUST be discovered — guards against a glob/regex change silently
# emptying the parametrization (a vacuous green).
_KNOWN_MODULES = {
    "WIRELM-SMD_ESP32-S3-WROOM-1",
    "WIFI-SMD_ESP32-WROOM-32E",
}


def _discover_antenna_footprints() -> list[tuple[str, str]]:
    """Return [(pretty_dir, footprint_name)] for antenna-bearing modules."""
    found: list[tuple[str, str]] = []
    for mod in sorted(_PARTS_LIBRARY.glob("*/*.pretty/*.kicad_mod")):
        if _ANTENNA_NAME_RE.search(mod.stem):
            found.append((str(mod.parent), mod.stem))
    return found


_ANTENNA_FOOTPRINTS = _discover_antenna_footprints()


def test_discovery_finds_known_modules():
    """The parametrization must not be vacuous: both ESP32 modules are found."""
    names = {name for _, name in _ANTENNA_FOOTPRINTS}
    missing = _KNOWN_MODULES - names
    assert not missing, (
        f"antenna-footprint discovery missed {missing}; the keep-out invariant "
        f"below would vacuously pass. Found: {sorted(names)}"
    )


@pytest.mark.parametrize(
    "pretty_dir,name",
    _ANTENNA_FOOTPRINTS,
    ids=[name for _, name in _ANTENNA_FOOTPRINTS],
)
def test_antenna_footprint_has_rf_keepout(pretty_dir: str, name: str):
    fp = pcbnew.FootprintLoad(pretty_dir, name)
    assert fp is not None, f"failed to load {name} from {pretty_dir}"

    # --- courtyard covers the body (non-empty on the mounted side) ---
    crtyd_layer = (
        pcbnew.F_CrtYd if fp.GetLayer() == pcbnew.F_Cu else pcbnew.B_CrtYd
    )
    cbox = fp.GetCourtyard(crtyd_layer).BBox()
    assert cbox.GetWidth() > 0 and cbox.GetHeight() > 0, (
        f"{name}: empty courtyard — placer falls back to copper bbox and the "
        f"antenna keep-out area is unmodeled"
    )

    # --- >=1 footprint-internal rule-area keep-out over the antenna ---
    # ZONES has no GetCount(); it is iterable -> materialize with list().
    rule_areas = [z for z in list(fp.Zones()) if z.GetIsRuleArea()]
    assert rule_areas, f"{name}: no footprint-internal rule-area zone (antenna keep-out missing)"

    qualifying = [
        z
        for z in rule_areas
        if z.GetDoNotAllowTracks()
        and z.GetDoNotAllowCopperPour()
        and z.GetDoNotAllowFootprints()
    ]
    assert qualifying, (
        f"{name}: has rule-area zone(s) but none disallow tracks+copperpour+"
        f"footprints. An antenna keep-out must keep parts, copper pour and "
        f"traces out of the near-field. "
        f"Flags seen: "
        + ", ".join(
            f"[trk={z.GetDoNotAllowTracks()} pour={z.GetDoNotAllowCopperPour()} "
            f"fp={z.GetDoNotAllowFootprints()}]"
            for z in rule_areas
        )
    )
