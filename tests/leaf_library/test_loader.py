"""Tests for kicraft.leaf_library.loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.leaf_library.loader import (
    BrokenLeaf,
    LeafLibrary,
    LoadedLeaf,
    resolve_library_dir,
)
from kicraft.leaf_library.manifest import (
    Dependencies,
    HierarchicalLabel,
    Interface,
    Manifest,
    Provenance,
    compute_content_hash,
    dump_manifest,
)


def _populate_valid_leaf(leaf_dir: Path, name: str = "test-leaf") -> Manifest:
    """Write a leaf dir with all required files + a valid manifest."""
    leaf_dir.mkdir(parents=True, exist_ok=True)
    (leaf_dir / "leaf_routed.kicad_pcb").write_text("(footprint x)", encoding="utf-8")
    (leaf_dir / "metadata.json").write_text(
        json.dumps({"subcircuit_id": {"sheet_name": "X"}}), encoding="utf-8"
    )
    (leaf_dir / "solved_layout.json").write_text(
        json.dumps({"schema_version": "subcircuits.layout.v1"}),
        encoding="utf-8",
    )
    (leaf_dir / "schematic.kicad_sch").write_text("(kicad_sch x)", encoding="utf-8")
    (leaf_dir / "autoplacer_fragment.json").write_text(
        json.dumps({}), encoding="utf-8"
    )
    (leaf_dir / "bom.csv").write_text("ref,value\nU1,foo\n", encoding="utf-8")

    placeholder_hash = "sha256:" + "0" * 64
    m = Manifest(
        name=name,
        version="0.1.0",
        content_hash=placeholder_hash,
        description="test",
        interface=Interface(
            hierarchical_labels=[HierarchicalLabel(name="VBUS", direction="input")]
        ),
        refs=["U1"],
        dependencies=Dependencies(
            kicad_symbol_libs=["Device"],
            kicad_footprint_libs=["Resistor_SMD"],
            kicad_version_min="9.0.0",
        ),
        provenance=Provenance(
            source_project_stem="x",
            source_sheet_name="X",
            source_experiment_round=1,
            promoted_at="2026-05-17T00:00:00Z",
            kicad_version="9.0.0",
        ),
    )
    dump_manifest(m, leaf_dir)
    real_hash = compute_content_hash(leaf_dir)
    m = m.model_copy(update={"content_hash": real_hash})
    dump_manifest(m, leaf_dir)
    return m


def test_resolve_library_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("KICRAFT_LEAF_LIB", str(tmp_path / "custom"))
    assert resolve_library_dir() == tmp_path / "custom"


def test_resolve_library_dir_default(monkeypatch):
    monkeypatch.delenv("KICRAFT_LEAF_LIB", raising=False)
    assert resolve_library_dir() == Path.home() / ".kicraft" / "leaves"


def test_missing_library_dir_is_empty(tmp_path):
    lib = LeafLibrary(tmp_path / "does-not-exist")
    loaded, broken = lib.load_all()
    assert loaded == []
    assert broken == []


def test_valid_leaf_loads(tmp_path):
    _populate_valid_leaf(tmp_path / "test-leaf")
    lib = LeafLibrary(tmp_path)
    loaded, broken = lib.load_all()
    assert len(loaded) == 1
    assert broken == []
    assert isinstance(loaded[0], LoadedLeaf)
    assert loaded[0].slug == "test-leaf@0.1.0"


def test_missing_required_files_marks_broken(tmp_path):
    leaf_dir = tmp_path / "incomplete"
    leaf_dir.mkdir()
    (leaf_dir / "leaf_routed.kicad_pcb").write_text("x", encoding="utf-8")
    lib = LeafLibrary(tmp_path)
    loaded, broken = lib.load_all()
    assert loaded == []
    assert len(broken) == 1
    assert "missing required file" in broken[0].reason


def test_malformed_manifest_marks_broken(tmp_path):
    leaf_dir = tmp_path / "malformed"
    leaf_dir.mkdir()
    # Lay down required files but with a broken manifest
    for name in (
        "leaf_routed.kicad_pcb",
        "metadata.json",
        "solved_layout.json",
        "schematic.kicad_sch",
        "autoplacer_fragment.json",
        "bom.csv",
    ):
        (leaf_dir / name).write_text("x", encoding="utf-8")
    (leaf_dir / "manifest.json").write_text("{not valid json", encoding="utf-8")
    lib = LeafLibrary(tmp_path)
    loaded, broken = lib.load_all()
    assert loaded == []
    assert len(broken) == 1


def test_hash_mismatch_marks_broken(tmp_path):
    leaf_dir = tmp_path / "test-leaf"
    _populate_valid_leaf(leaf_dir)
    # Mutate a file post-promotion -- hash should no longer match.
    (leaf_dir / "bom.csv").write_text("ref,value\nU99,bad\n", encoding="utf-8")
    lib = LeafLibrary(tmp_path)
    result = lib.load_one(leaf_dir)
    assert isinstance(result, BrokenLeaf)
    assert "content_hash mismatch" in result.reason


def test_hash_cache_keyed_by_mtime(tmp_path, monkeypatch):
    """Re-loading the same leaf without filesystem changes uses the cache."""
    leaf_dir = tmp_path / "test-leaf"
    _populate_valid_leaf(leaf_dir)
    lib = LeafLibrary(tmp_path)
    # First load populates the cache.
    r1 = lib.load_one(leaf_dir)
    assert isinstance(r1, LoadedLeaf)
    # Tamper with the hash cache so we can prove the cached value is used.
    cached_mtime, _cached_hash = lib._hash_cache[leaf_dir]
    lib._hash_cache[leaf_dir] = (cached_mtime, "sha256:" + "f" * 64)
    r2 = lib.load_one(leaf_dir)
    # Now the cached hash disagrees with the manifest -> broken.
    assert isinstance(r2, BrokenLeaf)


def test_dir_name_must_match_manifest_name(tmp_path):
    leaf_dir = tmp_path / "wrong-name"
    _populate_valid_leaf(leaf_dir, name="right-name")
    lib = LeafLibrary(tmp_path)
    result = lib.load_one(leaf_dir)
    assert isinstance(result, BrokenLeaf)
    assert "does not match manifest name" in result.reason


def test_find_resolves_slug(tmp_path):
    _populate_valid_leaf(tmp_path / "test-leaf")
    lib = LeafLibrary(tmp_path)
    leaf = lib.find("test-leaf@0.1.0")
    assert leaf is not None
    assert leaf.manifest.version == "0.1.0"


def test_find_wrong_version_returns_none(tmp_path):
    _populate_valid_leaf(tmp_path / "test-leaf")
    lib = LeafLibrary(tmp_path)
    assert lib.find("test-leaf@9.9.9") is None


def test_find_missing_returns_none(tmp_path):
    lib = LeafLibrary(tmp_path)
    assert lib.find("nope@1.0.0") is None
