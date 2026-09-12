"""Routing-config unit tests: allowlist, atomic save, and the Settings overlay.

The file is the admin page's only persistence; it must round-trip the routing
knobs, ignore anything outside the allowlist (so a hand edit can never widen the
kill switch or daily/total ceilings), and take precedence over the environment
without tripping the env-vs-profile conflict check.
"""
from __future__ import annotations

import json

import pytest

from kicraft.server import routing_config
from kicraft.server.config import DESIGN_PROFILES, Settings


def _write(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_missing_file_is_empty(tmp_path):
    config = routing_config.load(tmp_path / "absent.json")
    assert config.active_profile == ""
    assert config.authoritative is False
    assert config.as_dict() == {}


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "routing.json"
    saved = routing_config.save(
        routing_config.RoutingConfig(
            active_profile="deepseek",
            max_tokens_per_call=4096,
            project_llm_budget_usd=0.10,
            design_reasoning_tokens=2048,
            design_temperature=0.25,
        ),
        path,
    )
    assert saved == path
    assert list(tmp_path.glob("*.tmp")) == []  # atomic: no temp left behind
    assert set(json.loads(path.read_text())) == routing_config.ALLOWED_KEYS
    loaded = routing_config.load(path)
    assert loaded == routing_config.RoutingConfig(
        active_profile="deepseek",
        max_tokens_per_call=4096,
        project_llm_budget_usd=0.10,
        design_reasoning_tokens=2048,
        design_temperature=0.25,
    )


def test_load_ignores_non_allowlisted_keys(tmp_path):
    path = tmp_path / "routing.json"
    _write(
        path,
        {
            "active_profile": "luna",
            "max_tokens_per_call": 2048,
            # Never honored: secrets and the hard safety ceilings are .env-only.
            "api_key": "sk-secret",
            "kill_switch": True,
            "daily_usd_ceiling": 9999,
            "total_usd_ceiling": 9999,
        },
    )
    config = routing_config.load(path)
    assert config.active_profile == "luna"
    assert config.max_tokens_per_call == 2048
    assert config.as_dict()["active_profile"] == "luna"
    assert set(config.as_dict()) <= routing_config.ALLOWED_KEYS


def test_unknown_active_profile_ignores_the_whole_file(tmp_path):
    path = tmp_path / "routing.json"
    _write(path, {"active_profile": "pro", "max_tokens_per_call": 2048})
    config = routing_config.load(path)
    assert config.active_profile == ""
    assert config.max_tokens_per_call is None


def test_invalid_json_is_ignored(tmp_path):
    path = tmp_path / "routing.json"
    path.write_text("{not json", encoding="utf-8")
    assert routing_config.load(path).authoritative is False


def test_invalid_behavior_values_are_dropped_but_profile_applies(tmp_path):
    path = tmp_path / "routing.json"
    _write(
        path,
        {
            "active_profile": "luna",
            "max_tokens_per_call": 0,
            "project_llm_budget_usd": -1,
            "design_reasoning_tokens": "lots",
            "design_temperature": 2.0,
        },
    )
    config = routing_config.load(path)
    assert config.active_profile == "luna"
    assert config.max_tokens_per_call is None
    assert config.project_llm_budget_usd is None
    assert config.design_reasoning_tokens is None
    assert config.design_temperature is None


@pytest.mark.parametrize(
    "config",
    [
        routing_config.RoutingConfig(active_profile="bogus"),
        routing_config.RoutingConfig(active_profile="luna", max_tokens_per_call=0),
        routing_config.RoutingConfig(
            active_profile="luna", max_tokens_per_call=routing_config.MAX_TOKENS_PER_CALL_CEILING + 1
        ),
        routing_config.RoutingConfig(active_profile="luna", design_temperature=1.5),
        routing_config.RoutingConfig(active_profile="luna", design_reasoning_tokens=-1),
    ],
)
def test_save_rejects_out_of_bounds_values(tmp_path, config):
    with pytest.raises(ValueError):
        routing_config.save(config, tmp_path / "routing.json")
    assert not (tmp_path / "routing.json").exists()


def test_routing_config_selects_profile_and_overlays_behavior(tmp_path, monkeypatch):
    path = tmp_path / "routing.json"
    routing_config.save(
        routing_config.RoutingConfig(
            active_profile="deepseek",
            max_tokens_per_call=2048,
            project_llm_budget_usd=0.05,
            design_reasoning_tokens=1024,
            design_temperature=0.3,
        ),
        path,
    )
    monkeypatch.setenv("KICRAFT_ROUTING_CONFIG", str(path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    settings = Settings.from_env(dotenv=False)
    assert settings.design_profile == "deepseek"
    assert settings.backend == "deepseek"
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.model == DESIGN_PROFILES["deepseek"]["model"]
    assert settings.max_tokens_per_call == 2048
    assert settings.project_llm_budget_usd == pytest.approx(0.05)
    assert settings.design_reasoning_tokens == 1024
    assert settings.design_temperature == pytest.approx(0.3)


def test_routing_config_wins_over_a_stale_env_model(tmp_path, monkeypatch):
    # The routing config is authoritative, so a stale .env KICRAFT_MODEL must not
    # trip the env-vs-profile conflict check (it exists for env-driven selection).
    path = tmp_path / "routing.json"
    routing_config.save(routing_config.RoutingConfig(active_profile="luna"), path)
    monkeypatch.setenv("KICRAFT_ROUTING_CONFIG", str(path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("KICRAFT_DESIGN_PROFILE", "luna")
    monkeypatch.setenv("KICRAFT_MODEL", "stale/not-the-profile-model")
    settings = Settings.from_env(dotenv=False)
    assert settings.design_profile == "luna"
    assert settings.model == DESIGN_PROFILES["luna"]["model"]


def test_routing_config_deepseek_profile_requires_its_key(tmp_path, monkeypatch):
    path = tmp_path / "routing.json"
    routing_config.save(routing_config.RoutingConfig(active_profile="deepseek"), path)
    monkeypatch.setenv("KICRAFT_ROUTING_CONFIG", str(path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="DEEPSEEK_API_KEY is not set"):
        Settings.from_env(dotenv=False)


def test_env_drives_profile_when_routing_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_ROUTING_CONFIG", str(tmp_path / "absent.json"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("KICRAFT_DESIGN_PROFILE", raising=False)
    monkeypatch.delenv("KICRAFT_MODEL", raising=False)
    monkeypatch.delenv("KICRAFT_PROVIDER_ORDER", raising=False)
    monkeypatch.delenv("KICRAFT_MAX_PRICE_PROMPT", raising=False)
    monkeypatch.delenv("KICRAFT_MAX_PRICE_COMPLETION", raising=False)
    assert Settings.from_env(dotenv=False).design_profile == "luna"
