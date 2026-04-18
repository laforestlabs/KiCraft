#!/usr/bin/env python3
"""Automatic experiment artifact cleanup for KiCad projects.

Three modes of operation:

  --before-run   Prepare for a fresh experiment run by archiving best/ and
                 removing transient state while preserving caches.

  --after-run    Trim intermediate artifacts after a successful run while
                 keeping the valuable solved results.

  --nuke         Delete the entire experiments directory for a full reset.

Common flags:

  --experiments-dir DIR   Override the default .experiments/ path.
  --dry-run               Print what would be deleted without deleting.

Discovery logic: walk up from this script to find a directory that
contains a *.kicad_pro file; that is the project root, and
.experiments/ lives directly beneath it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Project root discovery
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk up from this script location to find the KiCad project root."""
    candidate = Path(__file__).resolve().parent
    # Safety cap - do not walk above the filesystem root
    for _ in range(20):
        if any(candidate.glob("*.kicad_pro")):
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    raise RuntimeError(
        "Could not locate project root (no *.kicad_pro found above "
        f"{Path(__file__).resolve().parent})"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir_size(path: Path) -> int:
    """Return total size in bytes of a directory tree (0 if missing)."""
    if not path.exists():
        return 0
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
    except OSError:
        pass
    return total


def _file_size(path: Path) -> int:
    """Return size in bytes of a single file (0 if missing)."""
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _human(nbytes: int) -> str:
    """Format bytes as a human-friendly string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} TB"


class _Cleaner:
    """Accumulates removal operations and tracks freed space."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.freed: int = 0
        self.deleted_dirs: list[str] = []
        self.deleted_files: list[str] = []
        self.skipped: list[str] = []

    # -- high-level actions ---------------------------------------------------

    def remove_dir(self, path: Path) -> None:
        """Remove a directory tree."""
        if not path.exists():
            return
        size = _dir_size(path)
        label = str(path)
        if self.dry_run:
            print(f"  [dry-run] would delete dir  {label}  ({_human(size)})")
            self.deleted_dirs.append(label)
            self.freed += size
            return
        shutil.rmtree(path, ignore_errors=True)
        self.deleted_dirs.append(label)
        self.freed += size

    def remove_file(self, path: Path) -> None:
        """Remove a single file."""
        if not path.exists():
            return
        size = _file_size(path)
        label = str(path)
        if self.dry_run:
            print(f"  [dry-run] would delete file {label}  ({_human(size)})")
            self.deleted_files.append(label)
            self.freed += size
            return
        try:
            path.unlink()
        except OSError:
            self.skipped.append(label)
            return
        self.deleted_files.append(label)
        self.freed += size

    def rename_dir(self, src: Path, dst: Path) -> None:
        """Rename (move) a directory, removing dst first if it exists."""
        if not src.exists():
            return
        if dst.exists():
            self.remove_dir(dst)
        label = f"{src} -> {dst}"
        if self.dry_run:
            print(f"  [dry-run] would rename {label}")
            return
        src.rename(dst)
        print(f"  renamed {label}")

    # -- summary --------------------------------------------------------------

    def summary(self) -> None:
        n_dirs = len(self.deleted_dirs)
        n_files = len(self.deleted_files)
        prefix = "[dry-run] " if self.dry_run else ""
        print()
        print(f"{prefix}Cleanup summary:")
        print(f"  Directories removed : {n_dirs}")
        print(f"  Files removed       : {n_files}")
        print(f"  Space freed         : {_human(self.freed)}")
        if self.skipped:
            print(f"  Skipped (errors)    : {len(self.skipped)}")
            for s in self.skipped:
                print(f"    {s}")


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------

def _before_run(exp: Path, c: _Cleaner) -> None:
    """Pre-run cleanup: archive best/, purge transient state, keep caches."""
    print(f"=== before-run cleanup in {exp} ===")

    # Archive best/ -> best_previous/
    best = exp / "best"
    best_prev = exp / "best_previous"
    c.rename_dir(best, best_prev)

    # Directories to delete
    for name in (
        "frames",
        "rounds",
        "round_renders",
        "workers",
        "hierarchical_autoexperiment",
    ):
        c.remove_dir(exp / name)

    # Individual files to delete
    for name in (
        "run_status.json",
        "run_status.txt",
        "experiment.pid",
        "report.html",
        "hierarchical_summary.json",
        "llups.db",
        "llups.db-shm",
        "llups.db-wal",
    ):
        c.remove_file(exp / name)

    # Preserved (just informational)
    for name in ("seed_bank.json", "elite_configs.json", "best_config.json"):
        p = exp / name
        if p.exists():
            print(f"  preserved {p}")

    # Preserve subcircuits/*/solved_layout.json and metadata.json
    subcircuits = exp / "subcircuits"
    if subcircuits.is_dir():
        for sub in sorted(subcircuits.iterdir()):
            if not sub.is_dir():
                continue
            kept = []
            for keep_name in ("solved_layout.json", "metadata.json"):
                if (sub / keep_name).exists():
                    kept.append(keep_name)
            if kept:
                join_str = ", ".join(kept)
                print(f"  preserved {sub.name}/{join_str}")

    c.summary()


def _after_run(exp: Path, c: _Cleaner) -> None:
    """Post-run cleanup: trim intermediate subcircuit artifacts and bulk dirs."""
    print(f"=== after-run cleanup in {exp} ===")

    # Subcircuit trimming - keep only metadata.json, solved_layout.json,
    # layout.kicad_pcb inside each subcircuit directory.
    keep_names = {"metadata.json", "solved_layout.json", "layout.kicad_pcb"}
    subcircuits = exp / "subcircuits"
    if subcircuits.is_dir():
        for sub in sorted(subcircuits.iterdir()):
            if not sub.is_dir():
                continue
            for item in sorted(sub.iterdir()):
                if item.name in keep_names:
                    continue
                if item.is_dir():
                    c.remove_dir(item)
                else:
                    c.remove_file(item)

    # Directories to delete wholesale
    for name in (
        "frames",
        "round_renders",
        "workers",
        "hierarchical_parent_smoke",
        "hierarchical_freerouting_demo",
    ):
        c.remove_dir(exp / name)

    # Delete individual rounds/*.json files (summary lives in experiments.jsonl)
    rounds_dir = exp / "rounds"
    if rounds_dir.is_dir():
        for f in sorted(rounds_dir.iterdir()):
            if f.is_file() and f.suffix == ".json":
                c.remove_file(f)
        # Remove the directory itself if it is now empty
        try:
            if rounds_dir.is_dir() and not any(rounds_dir.iterdir()):
                c.remove_dir(rounds_dir)
        except OSError:
            pass

    c.summary()


def _nuke(exp: Path, c: _Cleaner) -> None:
    """Full reset: delete the entire experiments directory."""
    print(f"=== nuke: removing {exp} ===")
    c.remove_dir(exp)
    c.summary()
    action = "Would delete" if c.dry_run else "Deleted"
    print(f"\n{action} entire experiments directory: {exp}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Clean experiment artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --before-run              # prep for fresh run\n"
            "  %(prog)s --after-run --dry-run      # preview post-run trim\n"
            "  %(prog)s --nuke                     # full reset\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--before-run",
        action="store_true",
        help="Pre-run cleanup (archive best, purge transient state).",
    )
    mode.add_argument(
        "--after-run",
        action="store_true",
        help="Post-run trim of intermediate artifacts.",
    )
    mode.add_argument(
        "--nuke",
        action="store_true",
        help="Delete the entire experiments directory.",
    )

    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help=(
            "Path to the experiments directory. "
            "Default: <project_root>/.experiments/"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without actually deleting.",
    )

    args = parser.parse_args(argv)

    # Resolve experiments directory
    if args.experiments_dir is not None:
        exp = args.experiments_dir.resolve()
    else:
        exp = _find_project_root() / ".experiments"

    if not exp.exists():
        if args.nuke:
            print(f"Nothing to do -- {exp} does not exist.")
            return
        print(f"Experiments directory not found: {exp}")
        print("Nothing to clean.")
        return

    c = _Cleaner(dry_run=args.dry_run)

    if args.before_run:
        _before_run(exp, c)
    elif args.after_run:
        _after_run(exp, c)
    elif args.nuke:
        _nuke(exp, c)


if __name__ == "__main__":
    main()
