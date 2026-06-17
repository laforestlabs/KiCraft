"""Shared fixtures for the KiCraft abuse / vulnerability tests."""
from __future__ import annotations

import pytest

from kicraft.server.accounts import AccountStore


@pytest.fixture
def store(tmp_path) -> AccountStore:
    """A throwaway accounts store on tmp (never the prod DB)."""
    return AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
