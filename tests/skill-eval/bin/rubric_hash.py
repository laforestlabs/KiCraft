#!/usr/bin/env python3
"""Compute / verify the content hash of the skill-eval scoring rubric.

The hash is taken over the *parsed semantic content* of ``rubric.yaml`` with
``meta.sha256`` removed, then canonicalised as sorted, compact JSON. This means:

  * reordering keys, editing comments, or reflowing whitespace -> hash unchanged
  * changing any weight, anchor, gate, or band -> hash changes

So the hash is a tripwire for *semantic* edits to the rubric. ``RUBRIC.md`` is a
separate human-readable mirror and never affects the hash.

Run with the repo venv so PyYAML is importable::

    .venv/bin/python tests/skill-eval/bin/rubric_hash.py compute
    .venv/bin/python tests/skill-eval/bin/rubric_hash.py compute --write
    .venv/bin/python tests/skill-eval/bin/rubric_hash.py check

Other scripts import :func:`load_rubric` to get the parsed rubric plus its
verified hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

RUBRIC_PATH = Path(__file__).resolve().parent.parent / "rubric.yaml"
RUBRIC_MD_PATH = Path(__file__).resolve().parent.parent / "RUBRIC.md"


def compute_hash(content: dict) -> str:
    """Return the sha256 of the rubric content with ``meta.sha256`` stripped."""
    payload = json.loads(json.dumps(content))  # deep copy via round-trip
    meta = payload.get("meta")
    if isinstance(meta, dict):
        meta.pop("sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_rubric(path: Path = RUBRIC_PATH, *, verify: bool = True) -> dict:
    """Parse the rubric, attach ``_computed_sha256``, and optionally verify it.

    Raises ``SystemExit`` on a hash mismatch when ``verify`` is True so callers
    never score against a rubric whose stored hash is stale.
    """
    content = yaml.safe_load(path.read_text())
    computed = compute_hash(content)
    stored = (content.get("meta") or {}).get("sha256")
    content["_computed_sha256"] = computed
    if verify and stored not in (computed, "UNCOMPUTED", None):
        sys.exit(
            f"rubric hash mismatch: stored {stored} != computed {computed}\n"
            f"Run: rubric_hash.py compute --write  (after an intentional edit, "
            f"bump meta.version too)."
        )
    return content


def _write_stored_hash(path: Path, new_hash: str) -> None:
    """Replace the ``sha256:`` line in-place, preserving comments/formatting."""
    text = path.read_text()
    new_text, n = re.subn(
        r'(?m)^(\s*sha256:\s*)["\']?[^"\'\n]*["\']?\s*$',
        rf'\g<1>"{new_hash}"',
        text,
        count=1,
    )
    if n != 1:
        sys.exit("could not locate a single `sha256:` line to update in rubric.yaml")
    path.write_text(new_text)


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
            _write_stored_hash(RUBRIC_PATH, computed)
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
