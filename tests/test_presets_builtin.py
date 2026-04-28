"""Tests for the built-in DEFAULT preset.

The DEFAULT preset is computed from current source-of-truth defaults
(DEFAULT_CONFIG / CONFIG_SEARCH_SPACE / PLACEMENT_PARAMS) rather than
stored on disk. The GUI must not let users overwrite or delete it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def isolated_presets_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the presets module at an empty per-test JSON store."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    yield tmp_path / "kicraft" / "presets.json"


def test_default_preset_appears_in_list(isolated_presets_store):
    from kicraft.gui import presets as preset_store

    items = preset_store.list_presets()
    names = [p.name for p in items]
    assert "DEFAULT" in names

    default = next(p for p in items if p.name == "DEFAULT")
    assert default.builtin is True
    assert "_placement_config" in default.config
    assert "_mutation_bounds" in default.config


def test_default_preset_appears_first(isolated_presets_store):
    from kicraft.gui import presets as preset_store

    # Add a user preset that would sort before "DEFAULT" alphabetically
    # (capital A < D) so we know "DEFAULT" wins on builtin priority, not
    # alphabetical luck.
    preset_store.save_preset("AAA-test", {"foo": 1})

    items = preset_store.list_presets()
    assert items[0].name == "DEFAULT"
    assert items[0].builtin is True


def test_save_preset_rejects_builtin_name(isolated_presets_store):
    from kicraft.gui import presets as preset_store

    with pytest.raises(ValueError, match="DEFAULT"):
        preset_store.save_preset("DEFAULT", {"hostile": "config"})

    # And the on-disk store must not have been mutated.
    assert not isolated_presets_store.exists() or "hostile" not in isolated_presets_store.read_text()


def test_delete_preset_refuses_builtin_name(isolated_presets_store):
    from kicraft.gui import presets as preset_store

    assert preset_store.delete_preset("DEFAULT") is False

    # DEFAULT must still appear in list_presets afterwards.
    names = [p.name for p in preset_store.list_presets()]
    assert "DEFAULT" in names


def test_load_preset_returns_builtin_config(isolated_presets_store):
    from kicraft.autoplacer.config import CONFIG_SEARCH_SPACE
    from kicraft.gui import presets as preset_store

    config = preset_store.load_preset("DEFAULT")
    assert isinstance(config, dict)

    bounds = config.get("_mutation_bounds")
    assert isinstance(bounds, dict)
    # Every searchable knob must have an entry, with values matching the
    # current spec [min, max].
    for key, spec in CONFIG_SEARCH_SPACE.items():
        assert key in bounds, f"DEFAULT preset missing mutation bound for {key}"
        lo, hi = bounds[key]
        assert lo == spec["min"]
        assert hi == spec["max"]


def test_user_preset_named_default_is_shadowed(
    isolated_presets_store, monkeypatch: pytest.MonkeyPatch
):
    """If presets.json on disk somehow contains a DEFAULT entry (legacy
    or hand-edited), list_presets must not surface it -- the built-in
    always wins."""
    from kicraft.gui import presets as preset_store

    # Bypass save_preset's guard by writing the JSON file directly.
    isolated_presets_store.parent.mkdir(parents=True, exist_ok=True)
    isolated_presets_store.write_text(
        '[{"name": "DEFAULT", "config": {"hostile": true}, "notes": "", "created_at": ""}]'
    )

    items = preset_store.list_presets()
    defaults = [p for p in items if p.name == "DEFAULT"]
    assert len(defaults) == 1
    assert defaults[0].builtin is True
    # The hostile config must be filtered out.
    assert "hostile" not in defaults[0].config


def test_builtin_preset_reflects_current_source(isolated_presets_store):
    """DEFAULT preset is computed lazily, so updating CONFIG_SEARCH_SPACE
    in code (e.g. a future tuning) flows through automatically."""
    from kicraft.autoplacer.config import CONFIG_SEARCH_SPACE
    from kicraft.gui import presets as preset_store

    config = preset_store.load_preset("DEFAULT")
    assert config is not None
    bounds_count = len(config["_mutation_bounds"])
    assert bounds_count == len(CONFIG_SEARCH_SPACE)


def test_is_builtin_helper(isolated_presets_store):
    from kicraft.gui import presets as preset_store

    assert preset_store.is_builtin("DEFAULT") is True
    assert preset_store.is_builtin("my-custom-preset") is False
    assert preset_store.is_builtin("default") is False  # case-sensitive
