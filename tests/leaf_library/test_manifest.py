"""Tests for kicraft.leaf_library.manifest."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kicraft.leaf_library.manifest import (
    Dependencies,
    HierarchicalLabel,
    Interface,
    Manifest,
    Provenance,
    compute_content_hash,
    dump_manifest,
    load_manifest,
)


def _make_valid() -> Manifest:
    return Manifest(
        schema_version="1",
        name="usb-c-lipo-charger",
        version="1.2.0",
        content_hash="sha256:" + "0" * 64,
        description="USB-C charger.",
        tags=["power", "charger"],
        watch_out_for=None,
        interface=Interface(
            hierarchical_labels=[
                HierarchicalLabel(name="VBUS_IN", direction="input"),
                HierarchicalLabel(name="VBAT", direction="bidirectional"),
            ]
        ),
        refs=["U1", "C1", "R1"],
        dependencies=Dependencies(
            kicad_symbol_libs=["Regulator_Linear"],
            kicad_footprint_libs=["Package_TO_SOT_SMD"],
            kicad_version_min="9.0.0",
        ),
        provenance=Provenance(
            source_project_stem="llups",
            source_sheet_name="CHARGER",
            source_experiment_round=47,
            promoted_at="2026-05-17T14:23:00Z",
            kicad_version="9.0.0",
        ),
    )


def test_valid_manifest_round_trip(tmp_path):
    m = _make_valid()
    dump_manifest(m, tmp_path)
    loaded = load_manifest(tmp_path)
    assert loaded == m


def _mutated_payload(**overrides) -> dict:
    payload = _make_valid().model_dump(mode="json")
    payload.update(overrides)
    return payload


def test_invalid_name_format():
    with pytest.raises(ValidationError):
        Manifest.model_validate(_mutated_payload(name="Bad_Name"))


def test_invalid_version_not_semver():
    with pytest.raises(ValidationError):
        Manifest.model_validate(_mutated_payload(version="v1"))


def test_invalid_content_hash_prefix():
    with pytest.raises(ValidationError):
        Manifest.model_validate(_mutated_payload(content_hash="sha1:abc"))


def test_invalid_content_hash_length():
    with pytest.raises(ValidationError):
        Manifest.model_validate(_mutated_payload(content_hash="sha256:abc"))


def test_invalid_ref_with_suffix():
    with pytest.raises(ValidationError):
        Manifest.model_validate(_mutated_payload(refs=["U1A"]))


def test_invalid_hier_label_name():
    with pytest.raises(ValidationError):
        Interface(hierarchical_labels=[
            HierarchicalLabel(name="lower_case", direction="input"),
        ])


def test_invalid_hier_label_direction():
    # The Literal narrows the type, so pydantic raises ValidationError.
    with pytest.raises(ValidationError):
        HierarchicalLabel.model_validate({"name": "V1", "direction": "weird"})


def test_extra_fields_forbidden():
    payload = _make_valid().model_dump(mode="json")
    payload["unexpected"] = 1
    with pytest.raises(ValidationError):
        Manifest.model_validate(payload)


def test_pin_direction_matches_circuitchat_models():
    """PinDirection is redefined inside leaf_library; assert it stays in
    sync with the canonical definition in circuitchat.models."""
    from kicraft.circuitchat.models import PinDirection as CC_PinDirection
    from kicraft.leaf_library.manifest import PinDirection as LL_PinDirection
    from typing import get_args
    assert set(get_args(CC_PinDirection)) == set(get_args(LL_PinDirection))


def test_content_hash_excludes_manifest_itself(tmp_path):
    """Recomputed hash is stable across manifest edits to non-file fields."""
    leaf_dir = tmp_path / "x"
    leaf_dir.mkdir()
    (leaf_dir / "a.txt").write_text("hello", encoding="utf-8")
    (leaf_dir / "manifest.json").write_text(
        json.dumps({"ignored": "wrong"}), encoding="utf-8"
    )
    h1 = compute_content_hash(leaf_dir)
    (leaf_dir / "manifest.json").write_text(
        json.dumps({"different": "content"}), encoding="utf-8"
    )
    h2 = compute_content_hash(leaf_dir)
    assert h1 == h2


def test_content_hash_changes_with_file_content(tmp_path):
    leaf_dir = tmp_path / "x"
    leaf_dir.mkdir()
    (leaf_dir / "a.txt").write_text("hello", encoding="utf-8")
    h1 = compute_content_hash(leaf_dir)
    (leaf_dir / "a.txt").write_text("world", encoding="utf-8")
    h2 = compute_content_hash(leaf_dir)
    assert h1 != h2


def test_content_hash_stable_across_paths(tmp_path):
    """Hash depends on relative paths + bytes, not absolute paths."""
    a = tmp_path / "lib1" / "x"
    b = tmp_path / "lib2" / "x"
    for d in (a, b):
        d.mkdir(parents=True)
        (d / "a.txt").write_text("hello", encoding="utf-8")
        (d / "sub").mkdir()
        (d / "sub" / "b.txt").write_text("world", encoding="utf-8")
    assert compute_content_hash(a) == compute_content_hash(b)
