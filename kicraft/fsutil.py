"""Atomic file-write helper shared by every ``state.json`` writer.

``state.json`` is the sole IPC channel between the web app, the build worker,
and the design CLI (see CLAUDE.md): a truncate-then-write leaves a torn or
empty file if the writer crashes mid-write, and a concurrently-timed reader in
another process sees partial JSON — after which every ``_load_state`` fails
with JSONDecodeError and the design session is lost. Write-to-tmp +
``os.replace`` makes every persist all-or-nothing on POSIX.
"""
from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: Path | str, text: str,
                      encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically.

    Same-directory tmp file (rename must not cross filesystems), pid-suffixed
    so two processes writing the same target can't truncate each other's
    in-flight tmp, then ``os.replace`` so readers see either the old or the
    new content, never a partial write.
    """
    p = Path(path)
    tmp = p.with_name(f"{p.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)
