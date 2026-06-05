"""Guard: every shipped vendored parts bundle must load cleanly.

Regression context: the antenna-keepout footprint edits (PRs #10-#14) changed
the ESP32 `.kicad_mod` files without re-running `validate-part`, so both ESP32
module bundles silently failed `content_hash` validation and were skipped by the
loader. That quietly disabled the most-used MCU bundles, forcing every ESP32
design to re-resolve the module from LCSC through the BOM tool loop.

This test asserts the *real* vendored library (not a fixture) loads with zero
broken bundles, so a future footprint/symbol edit that forgets to recompute the
manifest hash fails CI instead of degrading silently.

It deliberately bypasses `conftest.py`'s autouse `mask_vendored_tier` fixture
(which points `vendored_parts_dir()` at an empty tmp dir) by deriving the real
vendored directory from the loader module's own location and iterating it
directly -- so it is hermetic w.r.t. the home/project/extra tiers.
"""
from __future__ import annotations

from pathlib import Path

import kicraft.parts_library.loader as loader_mod
from kicraft.parts_library.loader import BrokenPart, Tier, _iter_part_dirs, _load_one


def _real_vendored_dir() -> Path:
    """The shipped ``kicraft/parts_library/`` dir, independent of any
    monkeypatching of ``vendored_parts_dir`` (matches its real return value)."""
    return Path(loader_mod.__file__).resolve().parent


def test_vendored_bundles_all_load_clean():
    vendored = _real_vendored_dir()
    broken: list[tuple[str, str]] = []
    active_names: set[str] = set()
    for part_dir in _iter_part_dirs(vendored):
        result = _load_one(part_dir, Tier.VENDORED)
        if isinstance(result, BrokenPart):
            broken.append((part_dir.name, result.reason))
        else:
            active_names.add(result.manifest.name)

    assert not broken, (
        "vendored parts bundles failed validation (re-run "
        "`kicraft validate-part <dir> --update-hash` after editing bundle files):\n"
        + "\n".join(f"  {name}: {reason}" for name, reason in broken)
    )
    assert active_names, "expected at least one vendored bundle to load"


def test_esp32_module_bundles_present():
    """Lock in the specific regression: both ESP32 modules must be active."""
    vendored = _real_vendored_dir()
    active_names = {
        result.manifest.name
        for part_dir in _iter_part_dirs(vendored)
        if not isinstance(result := _load_one(part_dir, Tier.VENDORED), BrokenPart)
    }
    assert {"esp32-wroom-32e-n4", "esp32-s3-wroom-1"} <= active_names, (
        f"ESP32 module bundles missing/broken; active vendored = {sorted(active_names)}"
    )
