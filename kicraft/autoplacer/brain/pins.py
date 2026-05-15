"""Leaf pinning: lock a specific round's solution as the canonical state.

The composer is artifact-driven -- it reads each leaf's metadata.json,
solved_layout.json, and leaf_routed.kicad_pcb from disk and treats them as
rigid modules. Pinning is just "copy the round_NNNN_* snapshots over the
canonical names so the composer picks them up." The pins.json manifest is
the source of truth; ensure_applied() reconciles the on-disk state with it
and is safe to call repeatedly.

This module assumes step (2) of the snapshot work: round_NNNN_metadata.json
and round_NNNN_solved_layout.json exist alongside round_NNNN_leaf_*.kicad_pcb
in each leaf artifact dir.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PINS_FILENAME = "pins.json"
PINS_SCHEMA_VERSION = "pins.v1"

# Files that make up a leaf's canonical state. All three must be present in
# the round snapshot for a pin to be applicable.
_LEAF_CANONICAL_FILES = (
    ("leaf_routed.kicad_pcb", "leaf_routed"),
    ("metadata.json", "metadata"),
    ("solved_layout.json", "solved_layout"),
)


def _pins_path(experiments_dir: Path) -> Path:
    return Path(experiments_dir) / PINS_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_pins(experiments_dir: Path) -> dict[str, Any]:
    """Return the current pin manifest, or an empty manifest if none exists."""
    path = _pins_path(experiments_dir)
    if not path.exists():
        return {"schema_version": PINS_SCHEMA_VERSION, "pinned_leaves": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": PINS_SCHEMA_VERSION, "pinned_leaves": {}}
    if not isinstance(payload, dict):
        return {"schema_version": PINS_SCHEMA_VERSION, "pinned_leaves": {}}
    payload.setdefault("schema_version", PINS_SCHEMA_VERSION)
    pinned = payload.get("pinned_leaves")
    if not isinstance(pinned, dict):
        payload["pinned_leaves"] = {}
    return payload


def _write_pins(experiments_dir: Path, manifest: dict[str, Any]) -> None:
    path = _pins_path(experiments_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: pins.json is read by the GUI in 8 places (monitor,
    # pipeline_graph, node_detail) plus compose_subcircuits. Mid-write
    # reads would surface as JSONDecodeError or stale pin state.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    tmp_path.replace(path)


def _leaf_artifact_dir(experiments_dir: Path, leaf_key: str) -> Path:
    return Path(experiments_dir) / "subcircuits" / leaf_key


def _round_snapshot_files(
    leaf_dir: Path, round_num: int
) -> dict[str, Path]:
    """Return {canonical_filename: snapshot_path} for a round, if all exist."""
    prefix = f"round_{int(round_num):04d}"
    out: dict[str, Path] = {}
    for canonical, _suffix in _LEAF_CANONICAL_FILES:
        snapshot = leaf_dir / f"{prefix}_{canonical}"
        if not snapshot.exists():
            return {}
        out[canonical] = snapshot
    return out


def _round_all_snapshot_files(
    leaf_dir: Path, round_num: int
) -> dict[Path, Path]:
    """Return {snapshot_path: canonical_path} for ALL ``round_NNNN_*`` files
    in the leaf dir + ``renders/`` subdir, mapped to their canonical names.

    Includes the three core files plus every render PNG (and any other
    round-prefixed artifact) so downstream consumers that key off
    canonical filenames -- e.g. the monitor's leaf pin-card thumbnail
    reading ``renders/routed_front_all.png`` -- always see the pinned
    round's content. Without this, only the core ``leaf_routed.kicad_pcb``
    / ``metadata.json`` / ``solved_layout.json`` get pinned and the
    monitor surfaces a stale render from whichever round happened to
    write ``routed_front_all.png`` last.
    """
    prefix = f"round_{int(round_num):04d}_"
    out: dict[Path, Path] = {}
    for d in (leaf_dir, leaf_dir / "renders"):
        if not d.is_dir():
            continue
        for entry in d.iterdir():
            if not entry.is_file():
                continue
            name = entry.name
            if not name.startswith(prefix):
                continue
            canonical = d / name[len(prefix):]
            out[entry] = canonical
    return out


def list_available_rounds(experiments_dir: Path, leaf_key: str) -> list[int]:
    """Return sorted round numbers that have a complete snapshot for this leaf.

    Used by the GUI to populate the per-leaf round picker.
    """
    leaf_dir = _leaf_artifact_dir(experiments_dir, leaf_key)
    if not leaf_dir.exists():
        return []
    rounds: set[int] = set()
    for entry in leaf_dir.iterdir():
        name = entry.name
        if not name.startswith("round_"):
            continue
        try:
            round_num = int(name.split("_", 2)[1])
        except (ValueError, IndexError):
            continue
        rounds.add(round_num)
    # Filter to rounds that have ALL three canonical files snapshotted.
    return sorted(r for r in rounds if _round_snapshot_files(leaf_dir, r))


def pin_leaf(
    experiments_dir: Path,
    leaf_key: str,
    round_num: int,
    *,
    source: str = "manual",
) -> dict[str, Any]:
    """Pin a leaf to a specific round. Applies the snapshot immediately.

    Raises FileNotFoundError if the round snapshot is incomplete.
    """
    leaf_dir = _leaf_artifact_dir(experiments_dir, leaf_key)
    snapshot = _round_snapshot_files(leaf_dir, round_num)
    if not snapshot:
        raise FileNotFoundError(
            f"round {round_num} for leaf {leaf_key} has no complete snapshot "
            f"in {leaf_dir} (need round_{round_num:04d}_leaf_routed.kicad_pcb, "
            f"round_{round_num:04d}_metadata.json, "
            f"round_{round_num:04d}_solved_layout.json)"
        )

    # Apply the snapshot now -- copy each round_NNNN_* file over the
    # canonical name. Composer will see the pinned state on its next load.
    # shutil.copy (not copy2) so the destination mtime is set to NOW: any
    # downstream mtime-based cache (canvas PNG, routed_front_all.png) needs
    # to invalidate against this change. shutil.copy2 preserved the round
    # snapshot's mtime, which is often EARLIER than existing render caches
    # and would silently leave stale PNGs in place.
    #
    # Pin EVERYTHING with the round prefix, not just the core three files:
    # the monitor's leaf pin-card thumbnail reads
    # ``renders/routed_front_all.png``, so leaving render PNGs unpinned
    # meant the monitor surfaced whichever round wrote that file last
    # (typically the final round of the leaves-only run, not the pinned
    # one) while the manual layout correctly showed the pinned content.
    #
    # Phase B made round_NNNN_* render PNGs hardlinks to their canonical
    # paths, so when the requested round is the one whose bytes are
    # already in canonical, ``shutil.copy`` raises SameFileError (src and
    # dst share an inode). In that case the canonical already IS the
    # pinned content; we still want canonical to bump its mtime to NOW
    # so downstream caches invalidate, so we unlink + re-link instead of
    # copying. For paths that ARE distinct (different inodes), use
    # ``os.replace`` after creating a fresh inode via a temp copy so the
    # write is atomic.
    for src, dst in _round_all_snapshot_files(leaf_dir, round_num).items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            src_stat = src.stat()
            dst_stat = dst.stat() if dst.exists() else None
        except OSError:
            src_stat, dst_stat = None, None
        same_inode = (
            src_stat is not None
            and dst_stat is not None
            and src_stat.st_dev == dst_stat.st_dev
            and src_stat.st_ino == dst_stat.st_ino
        )
        if same_inode:
            # Canonical already points at the pinned content via the
            # Phase B hardlink. Drop and re-link to bump mtime so caches
            # invalidate.
            try:
                dst.unlink()
                import os as _os
                _os.link(src, dst)
            except OSError:
                # Fall back to copy on filesystems that reject hardlinks.
                shutil.copy(src, dst)
        else:
            shutil.copy(src, dst)
    _invalidate_leaf_canvas_cache(leaf_dir)

    manifest = read_pins(experiments_dir)
    manifest["pinned_leaves"][leaf_key] = {
        "round": int(round_num),
        "source": source,
        "pinned_at": _now_iso(),
    }
    _write_pins(experiments_dir, manifest)
    return manifest["pinned_leaves"][leaf_key]


def unpin_leaf(experiments_dir: Path, leaf_key: str) -> bool:
    """Remove a leaf pin. Returns True if a pin was removed.

    Note: does NOT revert the canonical files -- the next leaf solve round
    will overwrite them naturally. We deliberately leave the on-disk state
    alone so an unpin doesn't surprise the user with a different layout.
    """
    manifest = read_pins(experiments_dir)
    if leaf_key not in manifest.get("pinned_leaves", {}):
        return False
    del manifest["pinned_leaves"][leaf_key]
    _write_pins(experiments_dir, manifest)
    return True


def list_pins(experiments_dir: Path) -> dict[str, dict[str, Any]]:
    """Return {leaf_key: pin_record} for all currently pinned leaves."""
    return dict(read_pins(experiments_dir).get("pinned_leaves", {}))


def required_leaf_status(experiments_dir: Path) -> dict[str, str]:
    """Discover leaves on disk and report each leaf's pin readiness.

    Used by the GUI to gate the "Start parent only" button. Status values:
    - "pinned":       leaf is in pins.json
    - "unpinned":     leaf has snapshots but no pin (blocks parent-only)
    - "no-snapshots": leaf exists but has nothing to pin (e.g. BT1 with
                       no internal nets); composer just uses canonical
                       state, so this does not block parent-only

    Returns empty dict if .experiments/subcircuits/ doesn't exist.
    """
    sub_root = Path(experiments_dir) / "subcircuits"
    if not sub_root.exists():
        return {}
    pinned = set(list_pins(experiments_dir).keys())
    out: dict[str, str] = {}
    for child in sub_root.iterdir():
        if not child.is_dir():
            continue
        # Skip parent_composition artifacts (named subcircuit__<hash>)
        if child.name.startswith("subcircuit__"):
            continue
        leaf_key = child.name
        if leaf_key in pinned:
            out[leaf_key] = "pinned"
        elif list_available_rounds(experiments_dir, leaf_key):
            out[leaf_key] = "unpinned"
        else:
            out[leaf_key] = "no-snapshots"
    return out


def parent_only_ready(experiments_dir: Path) -> tuple[bool, list[str]]:
    """Return (ready, blocking_leaf_keys).

    Parent-only is ready when every leaf with snapshots is pinned. Leaves
    with no snapshots are exempt (nothing to pin). Returns (False, []) if
    no leaves exist on disk yet.
    """
    statuses = required_leaf_status(experiments_dir)
    if not statuses:
        return False, []
    blockers = [k for k, s in statuses.items() if s == "unpinned"]
    return len(blockers) == 0, blockers


def is_pinned(experiments_dir: Path, leaf_key: str) -> int | None:
    """Return the pinned round number for this leaf, or None if not pinned."""
    pin = read_pins(experiments_dir).get("pinned_leaves", {}).get(leaf_key)
    if not isinstance(pin, dict):
        return None
    round_val = pin.get("round")
    return int(round_val) if isinstance(round_val, int) else None


def ensure_applied(experiments_dir: Path) -> dict[str, str]:
    """Make on-disk canonical files match every active pin.

    Idempotent. The composer calls this before loading artifacts so a pin
    is honored even if the latest leaf solve overwrote the canonical files.
    Returns {leaf_key: status} where status is "applied", "already-current",
    or "snapshot-missing".
    """
    manifest = read_pins(experiments_dir)
    statuses: dict[str, str] = {}
    for leaf_key, pin in manifest.get("pinned_leaves", {}).items():
        if not isinstance(pin, dict):
            continue
        round_num = pin.get("round")
        if not isinstance(round_num, int):
            continue
        leaf_dir = _leaf_artifact_dir(experiments_dir, leaf_key)
        snapshot = _round_snapshot_files(leaf_dir, round_num)
        if not snapshot:
            statuses[leaf_key] = "snapshot-missing"
            continue
        # Skip the copy if every round_NNNN_* snapshot file already
        # matches its canonical counterpart by SIZE. Round snapshots are
        # immutable so size match implies content match in practice.
        # (mtime comparison can't be used because pin_leaf uses
        # shutil.copy which sets dst mtime to NOW.)
        all_files = _round_all_snapshot_files(leaf_dir, round_num)
        all_current = True
        for src, dst in all_files.items():
            if not dst.exists() or src.stat().st_size != dst.stat().st_size:
                all_current = False
                break
        if all_current:
            statuses[leaf_key] = "already-current"
            continue
        for src, dst in all_files.items():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        _invalidate_leaf_canvas_cache(leaf_dir)
        statuses[leaf_key] = "applied"
    return statuses


def _invalidate_leaf_canvas_cache(leaf_dir: Path) -> None:
    """Delete the cached manual-layout canvas PNG + extent sidecar so the
    next ``render_leaf_canvas`` call re-renders from the just-pinned
    canonical PCB. Required because ``render_leaf_canvas``'s mtime check
    uses strict-less-than, so a same-second pin+canvas-mtime collision
    leaves the stale PNG in place -- the user then sees a different leaf
    in the manual layout than the pinned snapshot."""
    (leaf_dir / "renders" / "leaf_canvas.png").unlink(missing_ok=True)
    (leaf_dir / "renders" / "leaf_canvas.png.extent.json").unlink(missing_ok=True)
