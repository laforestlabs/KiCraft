from concurrent.futures import ThreadPoolExecutor
import os
import threading

import pytest

from kicraft.fsutil import atomic_write_text


def test_concurrent_writers_use_distinct_temporary_files(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    barrier = threading.Barrier(2, timeout=5)
    replace = os.replace
    sources = []

    def synchronized_replace(source, destination):
        sources.append(source)
        barrier.wait()
        replace(source, destination)

    monkeypatch.setattr(os, "replace", synchronized_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(atomic_write_text, target, str(i)) for i in range(2)]
        for future in futures:
            future.result()

    assert len(set(sources)) == 2
    assert target.read_text() in {"0", "1"}
    assert list(tmp_path.iterdir()) == [target]


def test_failed_replace_keeps_existing_file_and_cleans_own_temporary(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text("old")
    unrelated = tmp_path / "state.json.tmp-someone-else"
    unrelated.write_text("other")

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_text(target, "new")

    assert target.read_text() == "old"
    assert unrelated.read_text() == "other"
    assert set(tmp_path.iterdir()) == {target, unrelated}
