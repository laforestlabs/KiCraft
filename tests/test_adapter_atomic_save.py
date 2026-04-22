"""Regression tests for the atomic-save helper in the KiCad adapter.

These tests verify the four-step atomic save invariant
(write -> fsync -> rename -> dir fsync) without requiring pcbnew, by
using a fake board object that records its Save target.
"""

import os
from pathlib import Path

import pytest

from kicraft.autoplacer.hardware import adapter


class _FakeBoard:
    def __init__(self, payload: bytes = b"FAKE-PCB\n"):
        self._payload = payload
        self.save_calls: list[str] = []

    def Save(self, path: str) -> None:
        self.save_calls.append(path)
        with open(path, "wb") as f:
            f.write(self._payload)


def test_atomic_save_writes_final_path_and_no_temp_left(tmp_path: Path):
    out = tmp_path / "leaf.kicad_pcb"
    board = _FakeBoard(b"OK")

    adapter._atomic_save_board(board, str(out))

    assert out.exists()
    assert out.read_bytes() == b"OK"
    leftover = [p for p in tmp_path.iterdir() if p.name != out.name]
    assert leftover == [], f"unexpected leftover files: {leftover}"


def test_atomic_save_writes_to_temp_then_renames(tmp_path: Path):
    out = tmp_path / "leaf.kicad_pcb"
    board = _FakeBoard(b"PAYLOAD")

    adapter._atomic_save_board(board, str(out))

    # board.Save was called against a sibling temp path, never the final path
    assert len(board.save_calls) == 1
    save_target = board.save_calls[0]
    assert save_target != str(out)
    assert save_target.startswith(str(out) + ".stamp_tmp.")
    assert os.path.dirname(save_target) == str(tmp_path)


def test_atomic_save_overwrites_existing_file(tmp_path: Path):
    out = tmp_path / "leaf.kicad_pcb"
    out.write_bytes(b"OLD")
    board = _FakeBoard(b"NEW")

    adapter._atomic_save_board(board, str(out))

    assert out.read_bytes() == b"NEW"


def test_atomic_save_calls_fsync_on_file_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    fsync_calls: list[int] = []
    real_fsync = os.fsync

    def _spy_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(adapter.os, "fsync", _spy_fsync)

    out = tmp_path / "leaf.kicad_pcb"
    board = _FakeBoard(b"X")
    adapter._atomic_save_board(board, str(out))

    # Expect at least 2 fsync calls: one on the temp file, one on the dir.
    assert len(fsync_calls) >= 2


def test_atomic_save_with_no_directory_in_path_uses_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    board = _FakeBoard(b"REL")

    adapter._atomic_save_board(board, "leaf.kicad_pcb")

    out = tmp_path / "leaf.kicad_pcb"
    assert out.exists()
    assert out.read_bytes() == b"REL"
