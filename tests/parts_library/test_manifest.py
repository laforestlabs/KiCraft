"""Tests for kicraft.parts_library.manifest."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kicraft.parts_library import (
    PartManifest,
    Provenance,
    compute_content_hash,
    dump_manifest,
    load_manifest,
)

from .conftest import make_valid_manifest


def test_round_trip(tmp_path):
    m = make_valid_manifest()
    dump_manifest(m, tmp_path)
    loaded = load_manifest(tmp_path)
    assert loaded == m


def test_name_must_be_kebab():
    with pytest.raises(ValidationError):
        make_valid_manifest(name="Widget")  # uppercase rejected
    with pytest.raises(ValidationError):
        make_valid_manifest(name="1bad-start")  # leading digit rejected
    with pytest.raises(ValidationError):
        make_valid_manifest(name="trailing-")  # trailing dash rejected


def test_version_must_be_semver():
    with pytest.raises(ValidationError):
        m = make_valid_manifest()
        # Bypass model_copy validation by constructing fresh
        PartManifest(**{**m.model_dump(), "version": "1.0"})


def test_content_hash_format():
    base = make_valid_manifest().model_dump()
    with pytest.raises(ValidationError):
        PartManifest(**{**base, "content_hash": "deadbeef"})  # no sha256: prefix
    with pytest.raises(ValidationError):
        PartManifest(**{**base, "content_hash": "sha256:not-hex-here-xxx"})


def test_sourcing_key_format():
    base = make_valid_manifest().model_dump()
    # Valid vendor short-names
    PartManifest(**{**base, "sourcing": {"lcsc": "C1", "digikey": "x", "mouser": "y"}})
    # Invalid: uppercase, underscores, leading digit
    with pytest.raises(ValidationError):
        PartManifest(**{**base, "sourcing": {"LCSC": "x"}})
    with pytest.raises(ValidationError):
        PartManifest(**{**base, "sourcing": {"di_gi": "x"}})
    with pytest.raises(ValidationError):
        PartManifest(**{**base, "sourcing": {"1mouser": "x"}})


def test_content_hash_stable_across_layouts(tmp_path):
    """Reordering manifest write / file edits must change the hash; identical
    layouts in two different locations must produce the same hash."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    for root in (a, b):
        root.mkdir()
        (root / "widget.kicad_sym").write_text("(kicad_symbol_lib)")
        (root / "widget.pretty").mkdir()
        (root / "widget.pretty" / "F.kicad_mod").write_text("(footprint)")
    assert compute_content_hash(a) == compute_content_hash(b)

    # Mutating a file changes the hash.
    (a / "widget.kicad_sym").write_text("(kicad_symbol_lib (version 2))")
    assert compute_content_hash(a) != compute_content_hash(b)


def test_manifest_does_not_contribute_to_hash(tmp_path):
    """The hash must be stable across manifest rewrites since the manifest
    itself stores the hash — including itself would be circular."""
    (tmp_path / "widget.kicad_sym").write_text("(kicad_symbol_lib)")
    (tmp_path / "widget.pretty").mkdir()
    (tmp_path / "widget.pretty" / "F.kicad_mod").write_text("(footprint)")
    h1 = compute_content_hash(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"anything": "here"}))
    h2 = compute_content_hash(tmp_path)
    assert h1 == h2
