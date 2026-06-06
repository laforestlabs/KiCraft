#!/usr/bin/env python3
"""Compute / verify the content hash of the KiCraft scoring rubric.

Thin CLI over :mod:`kicraft.eval.rubric`, which is now the canonical home of the
hashing logic and of ``rubric.yaml`` itself (moved to ``kicraft/eval/`` so the
shipped web app can self-evaluate without importing anything under ``tests/``).
The hash is taken over the *parsed semantic content* with ``meta.sha256`` removed:

  * reordering keys, editing comments, or reflowing whitespace -> hash unchanged
  * changing any weight, anchor, gate, or band -> hash changes

``RUBRIC.md`` is a separate human-readable mirror (still in this directory) and
never affects the hash.

Run with the repo venv so kicraft + PyYAML are importable::

    .venv/bin/python tests/skill-eval/bin/rubric_hash.py compute
    .venv/bin/python tests/skill-eval/bin/rubric_hash.py compute --write
    .venv/bin/python tests/skill-eval/bin/rubric_hash.py check

Other scripts import :func:`load_rubric` (re-exported here) to get the parsed
rubric plus its verified hash.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# Canonical implementation + the moved rubric path live in the package now.
from kicraft.eval.rubric import (  # noqa: F401  (load_rubric re-exported for callers)
    RUBRIC_PATH,
    compute_hash,
    load_rubric,
    write_stored_hash,
)

RUBRIC_MD_PATH = Path(__file__).resolve().parent.parent / "RUBRIC.md"


def _md_in_sync(version, hash_: str) -> bool | None:
    """True/False if RUBRIC.md stamps the given version+hash; None if no stamp found."""
    if not RUBRIC_MD_PATH.exists():
        return None
    md = RUBRIC_MD_PATH.read_text()
    if hash_ not in md:
        return False
    return f"v{version}" in md or str(version) in md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute", help="print the computed hash")
    c.add_argument("--write", action="store_true", help="write it into rubric.yaml meta.sha256")
    sub.add_parser("check", help="verify stored hash matches content; exit 1 on mismatch")
    args = ap.parse_args(argv)

    content = yaml.safe_load(RUBRIC_PATH.read_text())
    meta = content.get("meta") or {}
    version = meta.get("version")
    stored = meta.get("sha256")
    computed = compute_hash(content)

    if args.cmd == "compute":
        print(f"rubric v{version}  sha256:{computed}")
        if args.write:
            write_stored_hash(RUBRIC_PATH, computed)
            print(f"wrote sha256 into {RUBRIC_PATH.name}")
        return 0

    # check
    ok = True
    if stored == "UNCOMPUTED" or stored is None:
        print(f"FAIL: meta.sha256 not set (computed {computed}). Run: compute --write")
        ok = False
    elif stored != computed:
        print(f"FAIL: stored {stored}\n      != computed {computed}")
        print("      The rubric content changed. If intentional: bump meta.version, then compute --write.")
        ok = False
    else:
        print(f"OK: rubric v{version} sha256:{computed} (stored hash matches content)")

    synced = _md_in_sync(version, computed)
    if synced is False:
        print(f"WARN: RUBRIC.md does not stamp the current v{version}/{computed[:12]} — regenerate it.")
    elif synced is None:
        print("WARN: RUBRIC.md not found — generate the human-readable mirror.")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
