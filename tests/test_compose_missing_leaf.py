"""Compose must FAIL LOUDLY when a child subcircuit produced no artifact.

NO FALLBACKS: a leaf whose solve failed yields no solved_layout.json artifact.
Composing the parent without it strands its components as loose parent-level
parts that get force/SA'd at the parent -- a fallback that cannot place a failed
leaf and only masks the failure. _missing_child_artifacts surfaces the dropped
children so main() can abort instead of degrading to that path.
"""
from __future__ import annotations

from types import SimpleNamespace

from kicraft.cli.compose_subcircuits import _missing_child_artifacts


def _cid(path: str, sheet: str):
    return SimpleNamespace(instance_path=path, sheet_name=sheet)


def _artifact(path: str):
    return SimpleNamespace(
        layout=SimpleNamespace(subcircuit_id=SimpleNamespace(instance_path=path))
    )


def _parent(*child_ids):
    return SimpleNamespace(child_ids=list(child_ids))


def test_all_children_present_no_missing() -> None:
    parent = _parent(_cid("/HEADER", "HEADER"), _cid("/LED_ARRAY", "LED ARRAY"))
    loaded = [_artifact("/HEADER"), _artifact("/LED_ARRAY")]
    assert _missing_child_artifacts(parent, loaded) == []


def test_dropped_leaf_is_reported() -> None:
    # LED_ARRAY solve failed -> no artifact -> must be reported as missing.
    parent = _parent(_cid("/HEADER", "HEADER"), _cid("/LED_ARRAY", "LED ARRAY"))
    loaded = [_artifact("/HEADER")]
    missing = _missing_child_artifacts(parent, loaded)
    assert [m.sheet_name for m in missing] == ["LED ARRAY"]


def test_no_parent_definition_requires_nothing() -> None:
    assert _missing_child_artifacts(None, []) == []
