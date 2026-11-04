"""Atomic file-write helper shared by every ``state.json`` writer.

``state.json`` is the sole IPC channel between the web app, the build worker,
and the design CLI: a truncate-then-write leaves a torn or
empty file if the writer crashes mid-write, and a concurrently-timed reader in
another process sees partial JSON — after which every ``_load_state`` fails
with JSONDecodeError and the design session is lost. Write-to-tmp +
``os.replace`` makes every persist all-or-nothing on POSIX.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path


def atomic_write_text(path: Path | str, text: str,
                      encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically.

    A unique same-directory tmp file avoids collisions between both processes
    and threads. ``os.replace`` then makes the new contents visible at once.
    """
    p = Path(path)
    while True:
        tmp = p.with_name(f"{p.name}.tmp-{secrets.token_hex(16)}")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            break
        except FileExistsError:
            continue
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)
