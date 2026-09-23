"""Focused boundaries for immutable external evaluation corpora."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kicraft.eval.external_briefs import load_external_manifest, validate_external_bundle


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "corpus_id": "external-fixtures-v1",
        "sampling": {"kind": "independent", "description": "Independently curated board briefs."},
        "capability_envelope": {"board_types": ["two-layer"], "max_area_mm2": 2500},
        "policy": {
            "max_cost_usd": 2.5,
            "max_duration_s": 600,
            "max_park_rounds": 3,
            "build_timeout_s": 180,
        },
        "briefs": [
            {
                "slug": "power-monitor",
                "brief": "Design a compact power monitor with a screw-terminal input.",
                "archetype": "power",
                "families": ["power", "connectors"],
                "eligible": True,
                "provenance": {"source": "independent review", "revision": "2026-09-22"},
            },
            {
                "slug": "held-out-example",
                "brief": "Reserve this independently curated brief for a future run.",
                "archetype": "control",
                "families": ["control"],
                "eligible": False,
                "provenance": {"source": "independent review"},
            },
        ],
    }


def _obligations() -> dict:
    return {
        "schema_version": 1,
        "corpus_id": "external-fixtures-v1",
        "briefs": {
            "power-monitor": [
                {
                    "id": "power-monitor.input-terminal",
                    "statement": "The input uses a physical screw terminal.",
                    "owner": "wiring/synthesis",
                    "evidence_source": "schematic/board connectivity artifacts",
                    "check": {
                        "kind": "part_class_count",
                        "part_class": "screw_terminal",
                        "minimum": 1,
                    },
                }
            ],
            "held-out-example": [],
        },
    }


def _write(tmp_path: Path, manifest: dict, obligations: dict) -> tuple[Path, Path]:
    manifest_path = tmp_path / "manifest.json"
    obligations_path = tmp_path / "obligations.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    obligations_path.write_text(json.dumps(obligations), encoding="utf-8")
    return manifest_path, obligations_path


def test_load_external_manifest_normalizes_and_revalidates_stored_bundle(tmp_path):
    manifest_path, obligations_path = _write(tmp_path, _manifest(), _obligations())

    bundle = load_external_manifest(manifest_path, obligations_path)

    assert bundle["entries"] == [
        {"index": 1, **_manifest()["briefs"][0]},
        {"index": 2, **_manifest()["briefs"][1]},
    ]
    assert bundle["hash"].startswith("sha256:")
    assert json.loads(json.dumps(bundle)) == bundle
    assert validate_external_bundle(copy.deepcopy(bundle)) == bundle


def test_external_bundle_hash_changes_for_manifest_policy_and_obligations(tmp_path):
    manifest = _manifest()
    obligations = _obligations()
    manifest_path, obligations_path = _write(tmp_path, manifest, obligations)
    original = load_external_manifest(manifest_path, obligations_path)["hash"]

    manifest["briefs"][0]["brief"] = "Design a compact power monitor with a fused input."
    _write(tmp_path, manifest, obligations)
    changed_brief = load_external_manifest(manifest_path, obligations_path)["hash"]

    manifest["policy"]["max_cost_usd"] = 3.0
    _write(tmp_path, manifest, obligations)
    changed_policy = load_external_manifest(manifest_path, obligations_path)["hash"]

    obligations["briefs"]["power-monitor"][0]["check"]["minimum"] = 2
    _write(tmp_path, manifest, obligations)
    changed_obligations = load_external_manifest(manifest_path, obligations_path)["hash"]

    assert len({original, changed_brief, changed_policy, changed_obligations}) == 4


@pytest.mark.parametrize("slug", ["../escape", "power/monitor"])
def test_external_manifest_rejects_path_escape_slug(tmp_path, slug):
    manifest = _manifest()
    manifest["briefs"][0]["slug"] = slug
    obligations = _obligations()
    obligations["briefs"] = {slug: obligations["briefs"]["power-monitor"], "held-out-example": []}
    manifest_path, obligations_path = _write(tmp_path, manifest, obligations)

    with pytest.raises(ValueError, match="safe lowercase slug"):
        load_external_manifest(manifest_path, obligations_path)


def test_external_manifest_rejects_duplicate_slug_and_missing_eligible_obligations(tmp_path):
    manifest = _manifest()
    manifest["briefs"][1]["slug"] = "power-monitor"
    obligations = _obligations()
    obligations["briefs"] = {"power-monitor": obligations["briefs"]["power-monitor"]}
    manifest_path, obligations_path = _write(tmp_path, manifest, obligations)

    with pytest.raises(ValueError, match="must be unique"):
        load_external_manifest(manifest_path, obligations_path)

    manifest = _manifest()
    obligations = _obligations()
    obligations["briefs"]["power-monitor"] = []
    manifest_path, obligations_path = _write(tmp_path, manifest, obligations)

    with pytest.raises(ValueError, match="eligible brief"):
        load_external_manifest(manifest_path, obligations_path)


def test_external_manifest_rejects_unknown_or_missing_obligation_slugs(tmp_path):
    obligations = _obligations()
    obligations["briefs"].pop("held-out-example")
    manifest_path, obligations_path = _write(tmp_path, _manifest(), obligations)

    with pytest.raises(ValueError, match="exactly the manifest slugs"):
        load_external_manifest(manifest_path, obligations_path)

    obligations = _obligations()
    obligations["briefs"]["substituted-brief"] = obligations["briefs"].pop("held-out-example")
    manifest_path, obligations_path = _write(tmp_path, _manifest(), obligations)

    with pytest.raises(ValueError, match="exactly the manifest slugs"):
        load_external_manifest(manifest_path, obligations_path)


def test_external_manifest_rejects_duplicate_obligation_ids(tmp_path):
    obligations = _obligations()
    obligations["briefs"]["held-out-example"] = [
        copy.deepcopy(obligations["briefs"]["power-monitor"][0])
    ]
    manifest_path, obligations_path = _write(tmp_path, _manifest(), obligations)

    with pytest.raises(ValueError, match="unique across the corpus"):
        load_external_manifest(manifest_path, obligations_path)


def test_external_manifest_rejects_nan_policy_and_invalid_check_shape(tmp_path):
    manifest = _manifest()
    manifest["policy"]["max_cost_usd"] = float("nan")
    manifest_path, obligations_path = _write(tmp_path, manifest, _obligations())

    with pytest.raises(ValueError, match="non-finite"):
        load_external_manifest(manifest_path, obligations_path)

    obligations = _obligations()
    obligations["briefs"]["power-monitor"][0]["check"] = {
        "kind": "part_class_count",
        "part_class": "screw_terminal",
    }
    manifest_path, obligations_path = _write(tmp_path, _manifest(), obligations)

    with pytest.raises(ValueError, match="minimum"):
        load_external_manifest(manifest_path, obligations_path)
