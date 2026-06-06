"""Load and content-hash the KiCraft evaluation rubric.

``rubric.yaml`` (alongside this module) is the canonical scoring contract: ten
anchored 0-4 dimensions, hard-fail gates, and grade bands. The hash is taken over
the *parsed semantic content* with ``meta.sha256`` removed, canonicalised as
sorted compact JSON, so reordering keys / editing comments / reflowing whitespace
leaves it unchanged, while changing any weight, anchor, gate, or band changes it.
Scores are only ever comparable within one hash.

This module is the single source of truth for both the offline skill-eval harness
(``tests/skill-eval/bin/``) and the in-app web self-evaluation (``kicraft.eval``).
The ``rubric_hash.py`` CLI is a thin wrapper over the functions here.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

RUBRIC_PATH = Path(__file__).resolve().parent / "rubric.yaml"


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
    content = yaml.safe_load(Path(path).read_text())
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


def write_stored_hash(path: Path, new_hash: str) -> None:
    """Replace the ``sha256:`` line in-place, preserving comments/formatting."""
    text = Path(path).read_text()
    new_text, n = re.subn(
        r'(?m)^(\s*sha256:\s*)["\']?[^"\'\n]*["\']?\s*$',
        rf'\g<1>"{new_hash}"',
        text,
        count=1,
    )
    if n != 1:
        sys.exit("could not locate a single `sha256:` line to update in rubric.yaml")
    Path(path).write_text(new_text)
