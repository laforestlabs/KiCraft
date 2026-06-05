"""Maturity badge: default, hash stability, vendored backfill, and promote-part."""
from __future__ import annotations

import argparse
from pathlib import Path

import kicraft.parts_library.loader as loader_mod
from kicraft.design.cli_app import _cmd_promote_part
from kicraft.parts_library import (
    compute_content_hash,
    dump_manifest,
    load_manifest,
    verify_content_hash,
)
from kicraft.parts_library.loader import BrokenPart, Tier, _iter_part_dirs, _load_one

from .conftest import make_valid_manifest, write_valid_part


def test_default_maturity_is_prototype():
    # conftest.make_valid_manifest does not set maturity -> the model default.
    assert make_valid_manifest().maturity == "prototype"


def test_maturity_edit_keeps_content_hash(tmp_path):
    part_dir = write_valid_part(tmp_path / "parts", name="widget")
    before = load_manifest(part_dir)
    assert before.maturity == "prototype"
    h = before.content_hash
    assert verify_content_hash(part_dir, before)
    # Editing only the badge must not change the content hash (excludes manifest.json).
    dump_manifest(before.model_copy(update={"maturity": "reviewed"}), part_dir)
    after = load_manifest(part_dir)
    assert after.maturity == "reviewed"
    assert after.content_hash == h
    assert compute_content_hash(part_dir) == h


def test_vendored_bundles_are_not_prototype():
    """Backfill guard: every shipped bundle is reviewed/production, not prototype."""
    vendored = Path(loader_mod.__file__).resolve().parent
    bad = []
    for d in _iter_part_dirs(vendored):
        r = _load_one(d, Tier.VENDORED)
        if not isinstance(r, BrokenPart) and r.manifest.maturity == "prototype":
            bad.append(d.name)
    assert not bad, f"vendored bundles still defaulting to prototype: {bad}"


def _promote_args(name, to, tier="home"):
    return argparse.Namespace(name=name, to=to, tier=tier)


def test_promote_prototype_to_reviewed(isolated_home):
    home_parts = isolated_home / ".kicraft" / "parts"
    part_dir = write_valid_part(home_parts, name="gizmo")
    assert load_manifest(part_dir).maturity == "prototype"
    assert _cmd_promote_part(_promote_args("gizmo", "reviewed")) == 0
    after = load_manifest(part_dir)
    assert after.maturity == "reviewed"
    assert verify_content_hash(part_dir, after)  # still valid (files unchanged)


def test_promote_to_production_requires_3d_model(isolated_home):
    home_parts = isolated_home / ".kicraft" / "parts"
    part_dir = write_valid_part(home_parts, name="gadget")
    # No 3D model -> refused, badge unchanged.
    assert _cmd_promote_part(_promote_args("gadget", "production")) == 2
    assert load_manifest(part_dir).maturity == "prototype"
    # Drop in a 3D model -> allowed, and the bundle stays valid (hash re-blessed
    # to cover the new file rather than going stale/broken).
    (part_dir / "3d").mkdir()
    (part_dir / "3d" / "gadget.step").write_text("dummy step content")
    assert _cmd_promote_part(_promote_args("gadget", "production")) == 0
    after = load_manifest(part_dir)
    assert after.maturity == "production"
    assert verify_content_hash(part_dir, after)


def test_promote_missing_bundle_errors(isolated_home):
    assert _cmd_promote_part(_promote_args("does-not-exist", "reviewed")) == 2


def test_add_part_files_path_stamps_maturity_into_home(isolated_home, tmp_path):
    """`add-part --into home --maturity reviewed` (files path) stamps the badge
    and lands the bundle in the home tier."""
    from kicraft.design import cli_app

    from .conftest import VALID_FOOTPRINT, VALID_SYMBOL

    sym = tmp_path / "w.kicad_sym"
    sym.write_text(VALID_SYMBOL)
    fp = tmp_path / "w.kicad_mod"
    fp.write_text(VALID_FOOTPRINT)
    rc = cli_app.main([
        "add-part", "--symbol", str(sym), "--footprint", str(fp),
        "--mpn", "WIDGET-1", "--name", "widgetpart",
        "--into", "home", "--maturity", "reviewed",
    ])
    assert rc == 0
    part_dir = isolated_home / ".kicraft" / "parts" / "widgetpart"
    m = load_manifest(part_dir)
    assert m.maturity == "reviewed"
    assert verify_content_hash(part_dir, m)
