import fcntl
import os

import pytest

from kicraft import build_slots


@pytest.mark.parametrize("failure", ["echo", "breadcrumb"])
def test_slot_is_released_when_acquisition_raises(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS_DIR", str(tmp_path))
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "1")

    def fail(*args):
        raise RuntimeError("injected failure")

    if failure == "breadcrumb":
        monkeypatch.setattr(os, "ftruncate", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        with build_slots.build_slot(echo=fail if failure == "echo" else None):
            pass

    fd = os.open(tmp_path / "slot_0.lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)
